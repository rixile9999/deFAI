package org.defai.udt;

import java.io.*;
import java.nio.channels.FileLock;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;

/** Transport-independent, durable resumable file exchange. No model execution. */
public final class Transfer {
    static final int MAGIC = 0x44465531;
    public static final int CHUNK = 65536;
    public static final long MAX_BYTES = 256L * 1024 * 1024;
    private Transfer() {}

    static MessageDigest digest() {
        try { return MessageDigest.getInstance("SHA-256"); }
        catch (NoSuchAlgorithmException e) { throw new IllegalStateException(e); }
    }
    static String hex(byte[] value) {
        StringBuilder result = new StringBuilder();
        for (byte b : value) result.append(String.format("%02x", b & 255));
        return result.toString();
    }
    public static String hash(File file, long limit) throws IOException {
        MessageDigest md = digest();
        try (InputStream in = new FileInputStream(file)) {
            byte[] buffer = new byte[CHUNK];
            while (limit > 0) {
                int n = in.read(buffer, 0, (int)Math.min(buffer.length, limit));
                if (n < 0) throw new EOFException("File changed or truncated");
                md.update(buffer, 0, n); limit -= n;
            }
        }
        return hex(md.digest());
    }
    static void header(DataOutputStream out, String id, String version, long size, String sha) throws IOException {
        out.writeInt(MAGIC); out.writeUTF(id); out.writeUTF(version);
        out.writeLong(size); out.writeUTF(sha); out.writeInt(CHUNK);
    }
    static void validate(String id, String version, long size, String sha) throws IOException {
        if (!id.matches("[A-Za-z0-9][A-Za-z0-9_-]{0,63}") ||
            !version.matches("[A-Za-z0-9][A-Za-z0-9_.-]{0,127}") ||
            size < 0 || size > MAX_BYTES || !sha.matches("[0-9a-f]{64}"))
            throw new IOException("Invalid transfer identity, model version, size or hash");
    }
    public static long send(File source, String id, String version, InputStream input, OutputStream output) throws IOException {
        if (!source.isFile()) throw new IOException("Source must be a regular file");
        long size = source.length();
        String sha = hash(source, size);
        validate(id, version, size, sha);
        DataInputStream in = new DataInputStream(input);
        DataOutputStream out = new DataOutputStream(new BufferedOutputStream(output, CHUNK));
        header(out, id, version, size, sha); out.flush();
        long offset = in.readLong();
        String prefix = in.readUTF();
        if (offset < 0 || offset > size || (offset != size && offset % CHUNK != 0) ||
            !hash(source, offset).equals(prefix)) throw new IOException("Resume prefix mismatch");
        long resumed = offset;
        try (RandomAccessFile file = new RandomAccessFile(source, "r")) {
            file.seek(offset);
            byte[] buffer = new byte[CHUNK];
            while (offset < size) {
                int count = (int)Math.min(CHUNK, size - offset);
                file.readFully(buffer, 0, count);
                MessageDigest md = digest(); md.update(buffer, 0, count);
                out.writeLong(offset); out.writeInt(count); out.write(md.digest());
                out.write(buffer, 0, count); out.flush();
                offset += count;
                if (in.readLong() != offset) throw new IOException("Invalid durable chunk acknowledgement");
            }
        }
        if (source.length() != size || !hash(source, size).equals(sha))
            throw new IOException("Source changed during transfer");
        out.writeLong(-1); out.flush();
        if (!"COMMITTED".equals(in.readUTF())) throw new IOException("Missing commit receipt");
        return resumed;
    }
    public static File receive(File root, InputStream input, OutputStream output) throws IOException {
        DataInputStream in = new DataInputStream(input);
        DataOutputStream out = new DataOutputStream(new BufferedOutputStream(output, CHUNK));
        if (in.readInt() != MAGIC) throw new IOException("Unsupported transfer version");
        String id = in.readUTF(), version = in.readUTF();
        long size = in.readLong(); String sha = in.readUTF();
        validate(id, version, size, sha);
        if (in.readInt() != CHUNK) throw new IOException("Unsupported chunk size");
        root = root.getCanonicalFile();
        if (!root.isDirectory() && !root.mkdirs()) throw new IOException("Cannot create receiver store");
        File directory = new File(root, id);
        if (!directory.getCanonicalFile().equals(directory)) throw new IOException("Transfer path redirects");
        if (!directory.isDirectory() && !directory.mkdir()) throw new IOException("Cannot create transfer directory");
        for (String name : new String[]{"lock", "manifest.bin", "payload.part", "payload.bin"})
            if (!new File(directory, name).getCanonicalFile().equals(new File(directory, name)))
                throw new IOException("Stored file redirects");
        try (RandomAccessFile lock = new RandomAccessFile(new File(directory, "lock"), "rw");
             FileLock held = lock.getChannel().tryLock()) {
            if (held == null) throw new IOException("Transfer already active");
            ByteArrayOutputStream bytes = new ByteArrayOutputStream();
            header(new DataOutputStream(bytes), id, version, size, sha);
            File manifest = new File(directory, "manifest.bin");
            if (manifest.exists()) {
                byte[] previous = new byte[(int)Math.min(1025, manifest.length())];
                try (DataInputStream stored = new DataInputStream(new FileInputStream(manifest))) { stored.readFully(previous); }
                if (!Arrays.equals(previous, bytes.toByteArray())) throw new IOException("Transfer ID conflicts with another model");
            } else {
                File temporary = File.createTempFile("manifest-", ".tmp", directory);
                try {
                    try (FileOutputStream stored = new FileOutputStream(temporary)) {
                        stored.write(bytes.toByteArray()); stored.getFD().sync();
                    }
                    if (!temporary.renameTo(manifest)) throw new IOException("Cannot publish manifest");
                } finally {
                    if (temporary.exists() && !temporary.delete()) temporary.deleteOnExit();
                }
            }
            File part = new File(directory, "payload.part"), complete = new File(directory, "payload.bin");
            boolean committed = complete.isFile();
            if (committed && (complete.length() != size || !hash(complete, size).equals(sha)))
                throw new IOException("Committed payload failed integrity check");
            try (RandomAccessFile file = new RandomAccessFile(committed ? complete : part, committed ? "r" : "rw")) {
                long offset = committed ? size : file.length();
                if (offset > size) throw new IOException("Partial payload exceeds declared size");
                // A process may die in the middle of a chunk. Only whole chunks are resumable.
                if (!committed && offset != size) { offset -= offset % CHUNK; file.setLength(offset); }
                out.writeLong(offset); out.writeUTF(hash(committed ? complete : part, offset)); out.flush();
                file.seek(offset);
                while (true) {
                    long position = in.readLong();
                    if (position == -1) break;
                    if (committed || position != offset || offset >= size) throw new IOException("Invalid chunk position");
                    int length = in.readInt();
                    if (length != Math.min(CHUNK, size - offset)) throw new IOException("Invalid chunk length");
                    byte[] expected = new byte[32], buffer = new byte[length];
                    in.readFully(expected); in.readFully(buffer);
                    if (!MessageDigest.isEqual(expected, digest().digest(buffer))) throw new IOException("Chunk checksum mismatch");
                    file.write(buffer); file.getFD().sync(); offset += length;
                    out.writeLong(offset); out.flush();
                }
                if (offset != size) throw new IOException("Premature transfer completion");
            }
            if (!committed) {
                if (!hash(part, size).equals(sha)) throw new IOException("Final checksum mismatch");
                if (!part.renameTo(complete)) throw new IOException("Cannot publish completed payload");
            }
            out.writeUTF("COMMITTED"); out.flush();
            return complete;
        }
    }
}
