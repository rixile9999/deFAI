package org.defai.udt;

import java.io.File;
import java.net.InetAddress;
import java.util.logging.Level;
import java.util.logging.Logger;

/** Trusted local/lab transport only: UDT is not an authenticated encrypted channel. */
public final class Main {
    public static void main(String[] args) throws Exception {
        if (!Boolean.getBoolean("defai.udt.verbose")) Logger.getLogger("udt").setLevel(Level.OFF);
        if (args.length < 1) throw new IllegalArgumentException("receive STORE BIND PORT TIMEOUT_MS | send FILE ID MODEL_VERSION HOST PORT TIMEOUT_MS [--allow-insecure-network]");
        if (args[0].equals("receive") && (args.length == 5 || args.length == 6)) {
            checkAddress(args[2], args);
            try (Session session = new Session(args[2], Integer.parseInt(args[3]), Long.parseLong(args[4]))) {
                System.out.println("READY " + session.port()); System.out.flush();
                session.accept();
                File result = Transfer.receive(new File(args[1]), session.input(), session.output());
                session.drain();
                System.out.println("COMMITTED " + result.getCanonicalPath());
            }
        } else if (args[0].equals("send") && (args.length == 7 || args.length == 8)) {
            checkAddress(args[4], args);
            try (Session session = new Session("0.0.0.0", 0, Long.parseLong(args[6]))) {
                session.connect(args[4], Integer.parseInt(args[5]));
                long offset = Transfer.send(new File(args[1]), args[2], args[3], session.input(), session.output());
                session.drain();
                System.out.println("COMMITTED resumed_bytes=" + offset);
            }
        } else throw new IllegalArgumentException("Invalid command arguments");
    }
    static void checkAddress(String address, String[] args) throws Exception {
        if (!InetAddress.getByName(address).isLoopbackAddress() && !args[args.length-1].equals("--allow-insecure-network"))
            throw new IllegalArgumentException("Non-loopback lab use requires --allow-insecure-network; no authentication/encryption is provided");
    }
}
