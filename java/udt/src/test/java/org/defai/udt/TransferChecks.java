package org.defai.udt;

import java.io.*;
import java.nio.file.Files;
import java.util.Arrays;

/** Stream-level adversarial cases, independent of UDP timing. */
public final class TransferChecks {
    static int checked;
    static byte[] wire(String id, String version, byte[] payload, int failure) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        DataOutputStream out = new DataOutputStream(bytes);
        String sha = Transfer.hex(Transfer.digest().digest(payload));
        Transfer.header(out, id, version, payload.length, failure == 2 ? new String(new char[64]).replace('\0', '0') : sha);
        for (int offset=0; failure!=5 && offset<payload.length; offset+=Transfer.CHUNK) {
            byte[] chunk = Arrays.copyOfRange(payload, offset, Math.min(offset+Transfer.CHUNK, payload.length));
            out.writeLong(failure == 3 ? offset+1 : offset);
            out.writeInt(failure == 4 ? Integer.MAX_VALUE : chunk.length);
            byte[] hash = Transfer.digest().digest(chunk);
            if (failure == 1) hash[0] ^= 1;
            out.write(hash); out.write(chunk);
        }
        out.writeLong(-1);
        return bytes.toByteArray();
    }
    static File receive(File root, byte[] bytes) throws IOException {
        return Transfer.receive(root, new ByteArrayInputStream(bytes), new ByteArrayOutputStream());
    }
    static void rejected(File root, byte[] bytes) throws IOException {
        try { receive(root, bytes); }
        catch (IOException expected) { checked++; return; }
        throw new AssertionError("Malformed transfer was accepted");
    }
    public static void main(String[] args) throws Exception {
        File root = new File(args[0]);
        byte[] payload = new byte[Transfer.CHUNK+17];
        for (int i=0; i<payload.length; i++) payload[i]=(byte)i;
        File complete = receive(root, wire("valid", "round-1", payload, 0));
        if (!Arrays.equals(payload, Files.readAllBytes(complete.toPath()))) throw new AssertionError("Payload mismatch");
        checked++;
        receive(root, wire("valid", "round-1", payload, 5)); checked++;
        receive(root, wire("empty", "round-1", new byte[0], 0)); checked++;
        rejected(root, wire("valid", "round-2", payload, 0));
        for (int failure=1; failure<=4; failure++) {
            rejected(root, wire("bad"+failure, "round-1", payload, failure));
            if (new File(root, "bad"+failure+"/payload.bin").exists()) throw new AssertionError("Bad bytes published");
        }
        rejected(root, wire("../escape", "round-1", payload, 0));
        rejected(root, wire("bad-version", "../round", payload, 0));
        byte[] truncated = wire("truncated", "round-1", payload, 0);
        rejected(root, Arrays.copyOf(truncated, truncated.length-20));
        byte[] magic = wire("magic", "round-1", payload, 0); magic[0] ^= 1;
        rejected(root, magic);
        Files.write(complete.toPath(), new byte[]{1,2,3});
        rejected(root, wire("valid", "round-1", payload, 0));
        File source = new File(root, "sender.bin"); Files.write(source.toPath(), payload);
        ByteArrayOutputStream reply = new ByteArrayOutputStream();
        DataOutputStream offset = new DataOutputStream(reply); offset.writeLong(Transfer.CHUNK); offset.writeUTF("wrong-prefix");
        try { Transfer.send(source, "prefix", "round-1", new ByteArrayInputStream(reply.toByteArray()), new ByteArrayOutputStream()); throw new AssertionError(); }
        catch (IOException expected) { checked++; }
        System.out.println(checked + " checks passed");
    }
}
