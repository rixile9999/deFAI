package org.defai.udt;

import java.io.*;
import java.net.*;
import java.util.concurrent.*;
import udt.*;

/** Bounded UDT session. Call on a worker thread on Android, never the UI thread. */
public final class Session implements AutoCloseable {
    private static final class EndPoint extends UDPEndPoint {
        EndPoint(DatagramSocket socket) { super(socket); }
        UDTSession take() throws InterruptedException { return accept(100, TimeUnit.MILLISECONDS); }
    }
    private final EndPoint endpoint;
    private volatile UDTSocket socket;
    private final ScheduledExecutorService timer;
    private final ScheduledFuture<?> deadline;
    private volatile boolean expired;

    public Session(String bind, int port, long timeoutMillis) throws IOException {
        if (timeoutMillis < 100 || timeoutMillis > 3600000) throw new IllegalArgumentException("Timeout must be 100..3600000 ms");
        endpoint = new EndPoint(new DatagramSocket(new InetSocketAddress(bind, port)));
        Thread owner = Thread.currentThread();
        timer = Executors.newSingleThreadScheduledExecutor(r -> { Thread t = new Thread(r, "defai-udt-deadline"); t.setDaemon(true); return t; });
        deadline = timer.schedule(() -> { expired = true; owner.interrupt(); closeSocket(); endpoint.stop(); }, timeoutMillis, TimeUnit.MILLISECONDS);
    }
    public void connect(String host, int port) throws Exception {
        UDTClient client = new UDTClient(endpoint);
        try { client.connect(host, port); }
        finally { socket = client.getEndpoint().getSessions().isEmpty() ? null : client.getEndpoint().getSessions().iterator().next().getSocket(); }
    }
    public void accept() throws Exception {
        endpoint.start(true);
        UDTSession accepted;
        do { accepted = endpoint.take(); } while (accepted == null);
        while (!accepted.isReady() || accepted.getSocket() == null) Thread.sleep(5);
        socket = accepted.getSocket();
    }
    public int port() { return endpoint.getLocalPort(); }
    public InputStream input() throws IOException {
        final UDTInputStream source = socket.getInputStream(); source.setBlocking(false);
        return new InputStream() {
            public int read() throws IOException { byte[] one = new byte[1]; return read(one, 0, 1) < 0 ? -1 : one[0] & 255; }
            public int read(byte[] bytes, int offset, int length) throws IOException {
                if (length == 0) return 0;
                byte[] buffer = new byte[length];
                while (true) {
                    if (expired || Thread.currentThread().isInterrupted()) throw new InterruptedIOException("UDT session deadline/cancellation");
                    int count = source.read(buffer);
                    if (count != 0) { if (count > 0) System.arraycopy(buffer, 0, bytes, offset, count); return count; }
                }
            }
        };
    }
    public OutputStream output() {
        // Stream flush publishes the application's bytes to the UDT send queue.
        // Transfer's durable chunk/commit receipts establish application delivery;
        // waiting for an additional transport ACK here can stall small replies.
        final OutputStream target = socket.getOutputStream();
        return new OutputStream() {
            public void write(int value) throws IOException { target.write(value); }
            public void write(byte[] bytes, int offset, int length) throws IOException { target.write(bytes, offset, length); }
            public void flush() {}
        };
    }
    public void drain() throws InterruptedException {
        // Keep receiving transport ACKs after the application commit receipt.
        // This is bounded by the same session deadline; retry uses durable ID.
        long rttMillis = socket.getSession().getStatistics().getRoundTripTimeMicros()/1000;
        Thread.sleep(Math.max(500, Math.min(5000, 4*rttMillis)));
    }
    private void closeSocket() {
        for (UDTSession session : endpoint.getSessions()) {
            UDTSocket current = session.getSocket();
            if (current != null) try { current.getReceiver().stop(); } catch (IOException ignored) { /* best effort after cancellation */ }
        }
    }
    public void close() {
        deadline.cancel(false); timer.shutdownNow(); closeSocket(); endpoint.stop();
        if (expired) Thread.interrupted();
    }
}
