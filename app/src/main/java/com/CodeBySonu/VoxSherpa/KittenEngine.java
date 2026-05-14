package com.CodeBySonu.VoxSherpa;

import android.content.Context;
import java.io.File;
import java.io.InputStream;
import java.io.FileOutputStream;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipEntry;

import com.k2fsa.sherpa.onnx.OfflineTts;
import com.k2fsa.sherpa.onnx.OfflineTtsConfig;
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig;
import com.k2fsa.sherpa.onnx.OfflineTtsKittenModelConfig;
import com.k2fsa.sherpa.onnx.GeneratedAudio;

/**
 * Third in-process voice family (alongside {@link VoiceEngine} for Piper
 * and {@link KokoroEngine} for Kokoro). Wraps sherpa-onnx's
 * {@link OfflineTtsKittenModelConfig} — the KittenTTS family
 * (Apache-2.0, ~25 MB int8 fp16 nano variant, 8 voices at 24 kHz).
 *
 * <p>Designed as the <em>smallest tier</em> in the storyvox voice picker
 * — a step below Piper-low for ultra-constrained devices (Raspberry Pi,
 * old phones, wearables) where even a 14 MB Piper voice is heavy. See
 * storyvox#119.</p>
 *
 * <p>Structurally a near-twin of {@link KokoroEngine}: shared multi-
 * speaker model (one ~25 MB ONNX + voices.bin + tokens.txt + the
 * espeak-ng-data already extracted by the Kokoro/Piper paths), speaker
 * picked at synth time via {@link #setActiveSpeakerId(int)}, generation
 * via the inherited sherpa-onnx {@link OfflineTts#generate} entry point.
 * The duplication is deliberate — both engines maintain their own
 * onnxruntime sessions so consumers can load one without paying the
 * other's memory cost. Storyvox's {@code EnginePlayer} dispatcher
 * decides which engine to talk to based on the {@code EngineType}
 * discriminator on the active voice.</p>
 *
 * <p><strong>What's the same as KokoroEngine:</strong> public no-arg
 * constructor for multi-instance parallelism, singleton {@link #getInstance()},
 * provider fallback (XNNPACK → CPU), espeak-ng-data extract path,
 * cancel cooperation via {@link #cancelRequested}, sample-rate accessor,
 * Sonic pitch-shift on top of PCM, smart-thread-count with explicit
 * override. The {@link #sonicQuality} static knob mirrors the field on
 * the other two engines for API parity (storyvox writes all three
 * together when the user flips the quality toggle).</p>
 *
 * <p><strong>What's different from KokoroEngine:</strong></p>
 * <ul>
 *   <li>No {@code phonemizerLang} static — Kitten's
 *       {@link OfflineTtsKittenModelConfig} doesn't expose a per-render
 *       language override (the nano-en model is English-only and uses
 *       its bundled phonemizer / espeak-ng-data path). If sherpa-onnx
 *       adds a {@code setLang}-equivalent to the Kitten config in a
 *       future release, mirror the Kokoro shape here.</li>
 *   <li>No {@code voiceLexicon} static — Kitten's config doesn't expose
 *       a {@code setLexicon} method either. Same future-proofing note as
 *       above.</li>
 *   <li>No silence-scale runtime setter — the Kitten model produces
 *       cleaner punctuation cadence at its baseline and we haven't seen
 *       the over-aggressive pause problem that pushed us to expose the
 *       knob on Kokoro. Add later if user feedback calls for it.</li>
 *   <li>Single config field {@code lengthScale} is fixed at 1.0; speed
 *       is driven via the per-generate {@code speed} arg the same way
 *       Piper does it (see {@link #generateAudioPCM}).</li>
 * </ul>
 *
 * <p><strong>Model file layout</strong> (storyvox-side shared dir, after
 * {@link #loadModel} resolves it):</p>
 * <pre>
 *   voices/_kitten_shared/
 *     model.onnx       (the fp16 ONNX weights)
 *     voices.bin       (speaker embeddings)
 *     tokens.txt       (phoneme vocabulary)
 *   files/espeak-ng-data/   (extracted from app assets — shared with
 *                            Piper + Kokoro)
 * </pre>
 *
 * <p>The shared-dir naming convention matches {@link KokoroEngine}'s
 * {@code voices/_kokoro_shared} — storyvox's {@code VoiceManager.kittenSharedDir()}
 * owns the actual file paths and hands them to {@link #loadModel}.</p>
 */
public class KittenEngine {

    static {
        try {
            System.loadLibrary("sherpa-onnx-jni");
        } catch (UnsatisfiedLinkError ignored) {}
    }

    /**
     * Sonic pitch-interpolation quality. Mirror of
     * {@link VoiceEngine#sonicQuality} and {@link KokoroEngine#sonicQuality};
     * see those fields' javadoc for the rationale and storyvox-side
     * wiring contract. Held as a separate field because the three
     * engines are independent classes with no shared base — storyvox
     * writes all three when the user flips the Settings toggle.
     */
    public static volatile int sonicQuality = 1;

    private static volatile KittenEngine instance;
    private OfflineTts tts;
    private String activeModelUri = "";
    private String activeTokensUri = "";
    private String activeVoicesBinUri = "";
    private Context activeContext = null;
    private String espeakDataPath = "";
    private int activeSpeakerId = 0;
    private volatile boolean cancelRequested = false;

    /**
     * Public constructor — enables multi-instance parallelism the same
     * way {@link VoiceEngine#VoiceEngine()} and {@link KokoroEngine#KokoroEngine()}
     * do. Each instance owns its own onnxruntime session; calls into
     * {@link #generateAudioPCM} are serialized per-instance via the
     * synchronized monitor on {@code this}. Two instances loaded against
     * the same model files → two independent OrtSessions, generate runs
     * in parallel.
     *
     * <p>Memory cost: each loaded Kitten session is small — the fp16
     * nano model is ~25 MB on disk and the runtime resident set hovers
     * around 60–80 MB once warmed. Even an 8-way fan-out fits in 1 GB,
     * which makes Kitten the friendliest engine for storyvox's Tier 3
     * (parallel synth) slider on low-end devices.</p>
     */
    public KittenEngine() {}

    // ── Singleton — thread-safe double-checked locking ───────────────────────
    public static KittenEngine getInstance() {
        if (instance == null) {
            synchronized (KittenEngine.class) {
                if (instance == null) {
                    instance = new KittenEngine();
                }
            }
        }
        return instance;
    }

    // ── Cancel ───────────────────────────────────────────────────────────────
    public void cancel() {
        cancelRequested = true;
    }

    // ── Smart thread count (mirrors VoiceEngine + KokoroEngine) ─────────────
    private int explicitNumThreads = 0;

    private int effectiveNumThreads() {
        return explicitNumThreads > 0 ? explicitNumThreads : getOptimalThreadCount();
    }

    private int getOptimalThreadCount() {
        int cores = Runtime.getRuntime().availableProcessors();
        if (cores >= 8) return 8;
        if (cores >= 6) return 6;
        if (cores >= 4) return 4;
        if (cores >= 2) return 2;
        return 1;
    }

    // ── espeak-ng-data extract ───────────────────────────────────────────────
    //
    // Identical extraction logic to KokoroEngine — Kitten reuses the
    // same espeak-ng-data bundle (sherpa-onnx's `kitten-nano-en-v0_1-fp16`
    // model is built against the same phonemizer data dir as Kokoro).
    // Lifted verbatim rather than refactored to a shared helper because
    // the engine classes intentionally have no shared base: each one is
    // a self-contained AAR-exported entry point.
    private synchronized void extractEspeakData(Context context) {
        if (context == null) return;
        File destDir = new File(context.getFilesDir(), "espeak-ng-data");
        String[] existing = destDir.list();

        if (!destDir.exists() || existing == null || existing.length == 0) {
            destDir.mkdirs();
            try (InputStream is = context.getAssets().open("espeak-ng-data.zip");
                 ZipInputStream zis = new ZipInputStream(is)) {

                ZipEntry ze;
                byte[] buffer = new byte[32768];
                while ((ze = zis.getNextEntry()) != null) {
                    File newFile = new File(destDir, ze.getName());
                    if (ze.isDirectory()) {
                        newFile.mkdirs();
                    } else {
                        File parent = newFile.getParentFile();
                        if (parent != null && !parent.exists()) parent.mkdirs();
                        try (FileOutputStream fos = new FileOutputStream(newFile)) {
                            int len;
                            while ((len = zis.read(buffer)) > 0) fos.write(buffer, 0, len);
                        }
                    }
                    zis.closeEntry();
                }
            } catch (Exception ignored) {}
        }

        File nestedDir = new File(destDir, "espeak-ng-data");
        espeakDataPath = new File(nestedDir, "phontab").exists()
                ? nestedDir.getAbsolutePath()
                : destDir.getAbsolutePath();
    }

    // ── Provider fallback: XNNPACK → CPU ────────────────────────────────────
    private OfflineTts createTtsWithFallback(String onnxPath, String tokensPath, String voicesBinPath) {
        String[] providers = {"xnnpack", "cpu"};

        for (String provider : providers) {
            try {
                if (cancelRequested) return null;

                OfflineTtsKittenModelConfig kittenConfig = new OfflineTtsKittenModelConfig();
                kittenConfig.setModel(onnxPath);
                kittenConfig.setTokens(tokensPath);
                kittenConfig.setVoices(voicesBinPath);
                kittenConfig.setDataDir(espeakDataPath);
                // lengthScale = 1.0f is the engine default and matches
                // Kokoro/Piper's behavior of driving speed via the
                // per-generate `speed` arg rather than baking it into
                // the model config. Storyvox's speed slider lands on
                // `generate(text, speakerId, speed)` directly.
                kittenConfig.setLengthScale(1.0f);

                OfflineTtsModelConfig modelConfig = new OfflineTtsModelConfig();
                modelConfig.setKitten(kittenConfig);
                modelConfig.setNumThreads(effectiveNumThreads());
                modelConfig.setProvider(provider);
                modelConfig.setDebug(false);

                OfflineTtsConfig config = new OfflineTtsConfig();
                config.setModel(modelConfig);
                // Match KokoroEngine's maxNumSentences=3 rather than
                // VoiceEngine's 5. Kitten is faster than Kokoro but
                // produces shorter natural-sounding spans before needing
                // a hard sentence break — three keeps the JNI batch
                // tight without truncating "Mr. Smith ran." style
                // mid-sentence periods.
                config.setMaxNumSentences(3);

                OfflineTts candidate = new OfflineTts(null, config);

                // Warm-up: generate a single space-tolerant token so we
                // catch a partial model file or wrong-arch native library
                // here at load time rather than mid-chapter. Mirrors
                // KokoroEngine's "..." probe and VoiceEngine's "Hello".
                GeneratedAudio test = candidate.generate("Hi.", activeSpeakerId, 1.0f);
                if (test != null && test.getSamples() != null && test.getSamples().length > 0) {
                    return candidate;
                }

                try { candidate.release(); } catch (Throwable ignored) {}

            } catch (Throwable ignored) {}
        }

        return null;
    }

    // ── Load model ───────────────────────────────────────────────────────────

    /**
     * v2.8.0 — overload that lets the caller override the numThreads
     * passed to sherpa-onnx. Pass {@code numThreads} = 0 for the auto
     * heuristic (same as the 4-arg {@link #loadModel}). Used by storyvox's
     * "Threads per engine" Settings slider; mirrors the same overload
     * on Kokoro/Piper.
     */
    public synchronized String loadModel(Context context, String onnxPath,
                                          String tokensPath, String voicesBinPath,
                                          int numThreads) {
        this.explicitNumThreads = numThreads;
        return loadModel(context, onnxPath, tokensPath, voicesBinPath);
    }

    public synchronized String loadModel(Context context, String onnxPath,
                                          String tokensPath, String voicesBinPath) {
        cancelRequested = false;

        // Avoid reloading if the exact same model is already active. Unlike
        // Kokoro, Kitten has no per-render language override so the cache key
        // is just the onnx path — speaker swaps reuse the loaded engine via
        // setActiveSpeakerId() with no reload.
        if (tts != null && activeModelUri.equals(onnxPath)) {
            return "Success";
        }

        if (onnxPath == null || onnxPath.isEmpty())           return "Error: ONNX path is empty.";
        if (tokensPath == null || tokensPath.isEmpty())       return "Error: Tokens path is empty.";
        if (voicesBinPath == null || voicesBinPath.isEmpty()) return "Error: voices.bin path is empty.";

        File fOnnx   = new File(onnxPath);
        File fTokens = new File(tokensPath);
        File fVoices = new File(voicesBinPath);

        if (!fOnnx.exists()   || fOnnx.length() == 0)   return "Error: ONNX file missing.";
        if (!fTokens.exists() || fTokens.length() == 0) return "Error: Tokens file missing.";
        if (!fVoices.exists() || fVoices.length() == 0) return "Error: voices.bin missing.";

        try {
            destroy();
            extractEspeakData(context);

            if (espeakDataPath == null || espeakDataPath.isEmpty())
                return "Error: espeak-ng-data extraction failed.";

            tts = createTtsWithFallback(onnxPath, tokensPath, voicesBinPath);

            if (tts == null) return "Error: Model load failed on all providers.";

            activeModelUri = onnxPath;
            activeTokensUri = tokensPath;
            activeVoicesBinUri = voicesBinPath;
            activeContext = context.getApplicationContext();
            return "Success";

        } catch (Throwable t) {
            activeModelUri = "";
            tts = null;
            return "Error: " + (t.getMessage() != null ? t.getMessage() : t.getClass().getSimpleName());
        }
    }

    // ── Generate audio PCM ───────────────────────────────────────────────────
    public byte[] generateAudioPCM(String inputText, float speedValue, float pitchValue) {
        if (cancelRequested) return null;
        if (inputText == null || inputText.trim().isEmpty()) return null;

        OfflineTts localTts;
        synchronized (this) {
            if (tts == null) return null;
            localTts = tts;
        }

        try {
            if (cancelRequested) return null;

            GeneratedAudio audio = localTts.generate(inputText.trim(), activeSpeakerId, speedValue);

            if (cancelRequested) return null;
            if (audio == null) return null;

            float[] audioFloats = audio.getSamples();
            if (audioFloats == null || audioFloats.length == 0) return null;

            short[] shortSamples = new short[audioFloats.length];
            for (int i = 0; i < audioFloats.length; i++) {
                float f = audioFloats[i];
                if (f > 1.0f) f = 1.0f;
                if (f < -1.0f) f = -1.0f;
                shortSamples[i] = (short) (f * 32767.0f);
            }

            // Sonic pitch shifting — identical shape to KokoroEngine /
            // VoiceEngine. The static `sonicQuality` knob lets storyvox
            // dial all three engines uniformly from a single Settings
            // toggle (see the KittenEngine#sonicQuality javadoc).
            if (pitchValue != 1.0f) {
                if (cancelRequested) return null;
                int sampleRate = localTts.sampleRate();
                if (sampleRate > 0) {
                    try {
                        com.CodeBySonu.VoxSherpa.Sonic sonic = new com.CodeBySonu.VoxSherpa.Sonic(sampleRate, 1);
                        sonic.setQuality(sonicQuality);
                        sonic.setPitch(pitchValue);
                        sonic.writeShortToStream(shortSamples, shortSamples.length);
                        sonic.flushStream();
                        int available = sonic.samplesAvailable();
                        if (available > 0) {
                            short[] outSamples = new short[available];
                            sonic.readShortFromStream(outSamples, available);
                            shortSamples = outSamples;
                        }
                    } catch (Throwable ignored) {
                        // Fall back to the un-pitched samples.
                    }
                }
            }

            if (cancelRequested) return null;

            // Short to PCM byte array — Little Endian (AudioTrack contract).
            byte[] pcmData = new byte[shortSamples.length * 2];
            for (int i = 0; i < shortSamples.length; i++) {
                pcmData[i * 2]     = (byte) (shortSamples[i] & 0xff);
                pcmData[i * 2 + 1] = (byte) ((shortSamples[i] >> 8) & 0xff);
            }

            return pcmData;

        } catch (Throwable t) {
            return null;
        }
    }

    // ── Sample rate ──────────────────────────────────────────────────────────
    public synchronized int getSampleRate() {
        if (tts == null) return 0;
        try {
            return tts.sampleRate();
        } catch (Throwable ignored) {
            return 0;
        }
    }

    // ── Speaker / Voice control ──────────────────────────────────────────────
    public void setActiveSpeakerId(int speakerId) {
        this.activeSpeakerId = speakerId;
    }

    public int getActiveSpeakerId() {
        return activeSpeakerId;
    }

    // ── State ────────────────────────────────────────────────────────────────
    public synchronized boolean isReady() {
        return tts != null;
    }

    public synchronized void destroy() {
        cancelRequested = false;
        if (tts != null) {
            try { tts.release(); } catch (Throwable ignored) {}
            tts = null;
            activeModelUri = "";
            activeTokensUri = "";
            activeVoicesBinUri = "";
            activeContext = null;
        }
    }
}
