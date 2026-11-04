package org.defai.udt;

import java.io.*;
import java.net.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import udt.*;

/** Long-running multi-session inbox; no shared model aggregation or HTTP path. */
public final class InboxServer {
    private static final class Listener extends UDPEndPoint {
        Listener(String bind, int port) throws IOException { super(new DatagramSocket(new InetSocketAddress(bind, port))); }
        UDTSession take() throws InterruptedException { return accept(100, TimeUnit.MILLISECONDS); }
    }
    public static void serve(File store, String bind, int port, long timeoutMillis, File lease) throws Exception {
        if (timeoutMillis < 1000 || timeoutMillis > 3600000) throw new IllegalArgumentException("Invalid session timeout");
        final Listener listener = new Listener(bind, port);
        final ThreadPoolExecutor workers = new ThreadPoolExecutor(16, 16, 0, TimeUnit.MILLISECONDS,
            new ArrayBlockingQueue<Runnable>(64), r -> { Thread t=new Thread(r,"defai-udt-inbox"); t.setDaemon(true); return t; });
        final AtomicBoolean running=new AtomicBoolean(true);
        Thread shutdown = new Thread(() -> {
            if (!running.getAndSet(false)) return;
            listener.stop(); workers.shutdownNow();
            for (UDTSession session : listener.getSessions()) {
                if (session.getSocket()!=null) try { session.getSocket().getReceiver().stop(); } catch (IOException ignored) {}
            }
        });
        Runtime.getRuntime().addShutdownHook(shutdown);
        ScheduledExecutorService watchdog=Executors.newSingleThreadScheduledExecutor(r -> { Thread t=new Thread(r,"defai-lease");t.setDaemon(true);return t; });
        if (lease!=null) watchdog.scheduleAtFixedRate(() -> {
            if(!lease.isFile() || System.currentTimeMillis()-lease.lastModified()>5000) shutdown.run();
        },500,500,TimeUnit.MILLISECONDS);
        listener.start(true);
        System.out.println("READY " + listener.getLocalPort()); System.out.flush();
        try {
            while (running.get() && !Thread.currentThread().isInterrupted()) {
                final UDTSession accepted = listener.take();
                if (accepted == null) continue;
                try {
                    workers.execute(() -> {
                        long deadline=System.nanoTime()+TimeUnit.MILLISECONDS.toNanos(timeoutMillis);
                        try {
                            while (!accepted.isReady() || accepted.getSocket()==null) {
                                if (System.nanoTime()>=deadline) throw new IOException("Handshake deadline");
                                Thread.sleep(5);
                            }
                            try (Session session=Session.attached(accepted.getSocket(),timeoutMillis)) {
                                Transfer.receive(store,session.input(),session.output());
                                session.drain();
                            }
                        } catch (Exception error) {
                            System.err.println("SESSION_FAILED " + error.getClass().getSimpleName());
                            if (accepted.getSocket()!=null) try { accepted.getSocket().getReceiver().stop(); } catch (IOException ignored) {}
                            listener.removeSession(accepted);
                            Thread.interrupted();
                        }
                    });
                } catch (RejectedExecutionException busy) {
                    if (accepted.getSocket()!=null) accepted.getSocket().getReceiver().stop();
                    listener.removeSession(accepted);
                }
            }
        } finally {
            shutdown.run();
            watchdog.shutdownNow();
            Runtime.getRuntime().removeShutdownHook(shutdown);
        }
    }
}
