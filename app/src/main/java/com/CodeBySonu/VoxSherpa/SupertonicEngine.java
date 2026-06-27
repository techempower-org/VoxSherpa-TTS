package com.CodeBySonu.VoxSherpa;

import android.content.Context;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

import com.k2fsa.sherpa.onnx.OfflineTts;
import com.k2fsa.sherpa.onnx.OfflineTtsConfig;
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig;
import com.k2fsa.sherpa.onnx.OfflineTtsSupertonicModelConfig;
import com.k2fsa.sherpa.onnx.GenerationConfig;
import com.k2fsa.sherpa.onnx.GeneratedAudio;

import kotlin.jvm.functions.Function1;

/**
 * Fourth in-process voice family (alongside {@link VoiceEngine} for Piper,
 * {@link KokoroEngine} for Kokoro and {@link KittenEngine} for Kitten).
 * Wraps sherpa-onnx's {@link OfflineTtsSupertonicModelConfig} — the
 * Supertonic 3 family (high-quality multi-speaker int8 model, 10 voices
 * at 24 kHz). See storyvox#1114.
 *
 * <p><strong>How Supertonic differs from the other three engines.</strong>
 * Kokoro/Kitten/Piper are phoneme models backed by an
 * {@code espeak-ng-data} directory and a single {@code model.onnx} +
 * {@code tokens.txt} (+ {@code voices.bin}). Supertonic 3 is a
 * unicode-tokenised, flow-matching architecture: it has NO espeak
 * dependency (tokenisation comes from {@link #FILE_UNICODE_INDEXER} +
 * {@link #FILE_TTS_JSON}) and ships as a <em>seven-file</em> bundle —
 * four ONNX graphs (duration predictor, text encoder, vector estimator,
 * vocoder) plus the JSON config, the unicode indexer and the
 * multi-speaker {@code voice.bin} style table. Because of that the load
 * surface here is directory-based ({@link #loadModel(Context, String)})
 * rather than the explicit (onnx, tokens, voicesBin) triple the other
 * engines take.</p>
 *
 * <p><strong>Generation.</strong> Supertonic runs through sherpa-onnx's
 * newer {@link OfflineTts#generateWithConfig} /
 * {@link OfflineTts#generateWithConfigAndCallback} entry points driven by
 * a {@link GenerationConfig} (speaker id, speed, and the flow-matching
 * step count {@link #numSteps}) rather than the legacy
 * {@code generate(text, sid, speed)} call. The blocking
 * {@link #generateAudioPCM} keeps the same byte[]-returning contract as
 * the other engines so storyvox's {@code EnginePlayer} dispatcher can
 * treat all four families uniformly; {@link #generateAudioPCMStreaming}
 * additionally exposes the per-chunk streaming callback for callers that
 * want first-audio latency below a full-sentence synth.</p>
 *
 * <p><strong>What's the same as the sibling engines:</strong> public
 * no-arg constructor for multi-instance parallelism, singleton
 * {@link #getInstance()}, provider fallback (XNNPACK → CPU), cancel
 * cooperation via {@link #cancelRequested}, sample-rate accessor, Sonic
 * pitch-shift on top of the PCM, smart-thread-count with an explicit
 * override. The {@link #sonicQuality} and {@link #voiceLexicon} static
 * knobs mirror the fields on the other engines for API parity (storyvox
 * writes them together when the user flips the relevant Settings toggle).</p>
 *
 * <p><strong>Model file layout</strong> (storyvox-side shared dir, the
 * argument to {@link #loadModel}):</p>
 * <pre>
 *   voices/_supertonic_shared/
 *     duration_predictor.int8.onnx
 *     text_encoder.int8.onnx
 *     vector_estimator.int8.onnx
 *     vocoder.int8.onnx
 *     tts.json
 *     unicode_indexer.bin
 *     voice.bin
 * </pre>
 * The canonical file names are exported as {@link #MODEL_FILES} so the
 * storyvox download + install-check code references one source of truth.
 */
public class SupertonicEngine {

    static {
        try {
            System.loadLibrary("sherpa-onnx-jni");
        } catch (UnsatisfiedLinkError ignored) {}
    }

    // ── Canonical model-file names (single source of truth) ─────────────────
    // Match sherpa-onnx's `sherpa-onnx-supertonic-3-tts-int8-*` bundle
    // layout. storyvox's VoiceManager references these constants for both
    // the download targets and the install-presence check so the two
    // sides can never drift.
    public static final String FILE_DURATION_PREDICTOR = "duration_predictor.int8.onnx";
    public static final String FILE_TEXT_ENCODER       = "text_encoder.int8.onnx";
    public static final String FILE_VECTOR_ESTIMATOR   = "vector_estimator.int8.onnx";
    public static final String FILE_VOCODER            = "vocoder.int8.onnx";
    public static final String FILE_TTS_JSON           = "tts.json";
    public static final String FILE_UNICODE_INDEXER    = "unicode_indexer.bin";
    public static final String FILE_VOICE_STYLE        = "voice.bin";

    /** All seven files that make up a Supertonic 3 install, in no
     *  particular order. Exposed so storyvox can iterate the bundle
     *  (download loop, presence check) without re-declaring the names. */
    public static final String[] MODEL_FILES = {
            FILE_DURATION_PREDICTOR,
            FILE_TEXT_ENCODER,
            FILE_VECTOR_ESTIMATOR,
            FILE_VOCODER,
            FILE_TTS_JSON,
            FILE_UNICODE_INDEXER,
            FILE_VOICE_STYLE,
    };

    /**
     * Sonic pitch-interpolation quality. Mirror of
     * {@link VoiceEngine#sonicQuality} / {@link KokoroEngine#sonicQuality} /
     * {@link KittenEngine#sonicQuality}; see those fields' javadoc for the
     * rationale and storyvox-side wiring contract. Held as a separate field
     * because the four engines are independent classes with no shared base —
     * storyvox writes all of them when the user flips the quality toggle.
     */
    public static volatile int sonicQuality = 1;

    /**
     * Per-voice lexicon override, included for API parity with
     * {@link KokoroEngine#voiceLexicon} so storyvox can write the same
     * static across every engine from one Settings code path.
     *
     * <p><strong>Currently inert for Supertonic.</strong> sherpa-onnx's
     * {@link OfflineTtsSupertonicModelConfig} (1.13.3) exposes no
     * {@code setLexicon} equivalent — the model tokenises via its bundled
     * {@link #FILE_UNICODE_INDEXER} + {@link #FILE_TTS_JSON}, not an
     * espeak lexicon. The field is read nowhere in this class; it exists
     * only so the storyvox "write all engines' lexicon together" path
     * compiles uniformly. If a future sherpa-onnx adds a lexicon hook to
     * the Supertonic config, wire it in {@link #createTtsWithFallback}
     * the way {@link KokoroEngine} does.</p>
     */
    public static volatile String voiceLexicon = "";

    /**
     * Flow-matching solver step count passed to
     * {@link GenerationConfig#setNumSteps}. Supertonic's vector estimator
     * is an ODE-solved flow-matching model; more steps = better prosody
     * at a linear latency cost. 8 matches sherpa-onnx's reference
     * {@code SupertonicTts} example and is a good quality/speed balance
     * for the int8 bundle. Tunable via {@link #setNumSteps} so storyvox
     * could surface a "synthesis quality" slider later.
     */
    public static final int DEFAULT_NUM_STEPS = 8;
    private volatile int numSteps = DEFAULT_NUM_STEPS;

    /** Language hint passed through {@link GenerationConfig#setExtra}
     *  under the {@code "lang"} key (sherpa-onnx Supertonic convention).
     *  The shipped bundle is English-only, so "en" is the only value
     *  storyvox uses today; kept as a field for symmetry with Kokoro's
     *  phonemizer-language override. */
    public static final String DEFAULT_LANG = "en";
    private volatile String lang = DEFAULT_LANG;

    private static volatile SupertonicEngine instance;
    private OfflineTts tts;
    private String activeModelDir = "";
    private int activeSpeakerId = 0;
    private volatile boolean cancelRequested = false;

    /**
     * Public constructor — enables multi-instance parallelism the same way
     * the sibling engines do. Each instance owns its own onnxruntime
     * session; calls into {@link #generateAudioPCM} are serialized
     * per-instance via the synchronized monitor on {@code this}. Two
     * instances loaded against the same model files → two independent
     * OrtSessions, generate runs in parallel.
     */
    public SupertonicEngine() {}

    // ── Singleton — thread-safe double-checked locking ───────────────────────
    public static SupertonicEngine getInstance() {
        if (instance == null) {
            synchronized (SupertonicEngine.class) {
                if (instance == null) {
                    instance = new SupertonicEngine();
                }
            }
        }
        return instance;
    }

    // ── Cancel ───────────────────────────────────────────────────────────────
    public void cancel() {
        cancelRequested = true;
    }

    // ── Smart thread count (mirrors the sibling engines) ────────────────────
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

    // ── Provider fallback: XNNPACK → CPU ────────────────────────────────────
    private OfflineTts createTtsWithFallback(
            String durationPredictor, String textEncoder, String vectorEstimator,
            String vocoder, String ttsJson, String unicodeIndexer, String voiceStyle) {
        String[] providers = {"xnnpack", "cpu"};

        for (String provider : providers) {
            try {
                if (cancelRequested) return null;

                OfflineTtsSupertonicModelConfig supertonicConfig =
                        new OfflineTtsSupertonicModelConfig();
                supertonicConfig.setDurationPredictor(durationPredictor);
                supertonicConfig.setTextEncoder(textEncoder);
                supertonicConfig.setVectorEstimator(vectorEstimator);
                supertonicConfig.setVocoder(vocoder);
                supertonicConfig.setTtsJson(ttsJson);
                supertonicConfig.setUnicodeIndexer(unicodeIndexer);
                supertonicConfig.setVoiceStyle(voiceStyle);

                OfflineTtsModelConfig modelConfig = new OfflineTtsModelConfig();
                modelConfig.setSupertonic(supertonicConfig);
                modelConfig.setNumThreads(effectiveNumThreads());
                modelConfig.setProvider(provider);
                modelConfig.setDebug(false);

                OfflineTtsConfig config = new OfflineTtsConfig();
                config.setModel(modelConfig);
                // Match the sibling engines' maxNumSentences=3: storyvox
                // batches sentences with internal periods ("Mr. Smith
                // ran.") and three keeps the JNI batch tight without
                // truncating mid-sentence periods.
                config.setMaxNumSentences(3);

                OfflineTts candidate = new OfflineTts(null, config);

                // Warm-up: synth a tiny utterance through the SAME
                // GenerationConfig path the real calls use (so a bad
                // numSteps / model file is caught here at load time, not
                // mid-chapter). Mirrors Kitten's "Hi." probe.
                GeneratedAudio test = candidate.generateWithConfig("Hi.", buildConfig(1.0f));
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
     * Load the Supertonic 3 bundle from {@code modelDir}, resolving the
     * seven {@link #MODEL_FILES} by their canonical names. Pass
     * {@code numThreads} = 0 for the auto heuristic (same as the 2-arg
     * overload). Used by storyvox's "Threads per engine" Settings slider;
     * mirrors the same overload on the sibling engines.
     */
    public synchronized String loadModel(Context context, String modelDir, int numThreads) {
        this.explicitNumThreads = numThreads;
        return loadModel(context, modelDir);
    }

    public synchronized String loadModel(Context context, String modelDir) {
        cancelRequested = false;

        // Avoid reloading if the exact same model dir is already active.
        // Speaker swaps reuse the loaded engine via setActiveSpeakerId()
        // with no reload — the speaker id is just a GenerationConfig knob.
        if (tts != null && activeModelDir.equals(modelDir)) {
            return "Success";
        }

        if (modelDir == null || modelDir.isEmpty()) return "Error: model dir is empty.";

        File dir = new File(modelDir);
        if (!dir.isDirectory()) return "Error: model dir missing: " + modelDir;

        File fDuration = new File(dir, FILE_DURATION_PREDICTOR);
        File fEncoder  = new File(dir, FILE_TEXT_ENCODER);
        File fVector   = new File(dir, FILE_VECTOR_ESTIMATOR);
        File fVocoder  = new File(dir, FILE_VOCODER);
        File fJson     = new File(dir, FILE_TTS_JSON);
        File fIndexer  = new File(dir, FILE_UNICODE_INDEXER);
        File fVoice    = new File(dir, FILE_VOICE_STYLE);

        for (File f : new File[]{fDuration, fEncoder, fVector, fVocoder, fJson, fIndexer, fVoice}) {
            if (!f.exists() || f.length() == 0) {
                return "Error: missing Supertonic model file: " + f.getName();
            }
        }

        try {
            destroy();

            tts = createTtsWithFallback(
                    fDuration.getAbsolutePath(),
                    fEncoder.getAbsolutePath(),
                    fVector.getAbsolutePath(),
                    fVocoder.getAbsolutePath(),
                    fJson.getAbsolutePath(),
                    fIndexer.getAbsolutePath(),
                    fVoice.getAbsolutePath());

            if (tts == null) return "Error: Model load failed on all providers.";

            activeModelDir = modelDir;
            return "Success";

        } catch (Throwable t) {
            activeModelDir = "";
            tts = null;
            return "Error: " + (t.getMessage() != null ? t.getMessage() : t.getClass().getSimpleName());
        }
    }

    // ── GenerationConfig builder ─────────────────────────────────────────────
    // Centralises the per-synth config so the warm-up probe, the blocking
    // generate and the streaming generate all use identical settings.
    private GenerationConfig buildConfig(float speed) {
        GenerationConfig gc = new GenerationConfig();
        gc.setSid(activeSpeakerId);
        gc.setSpeed(speed);
        gc.setNumSteps(numSteps);
        Map<String, String> extra = new HashMap<>();
        String l = lang;
        if (l != null && !l.isEmpty()) {
            extra.put("lang", l);
        }
        gc.setExtra(extra);
        return gc;
    }

    // ── Float[] → little-endian PCM16 (AudioTrack contract) ──────────────────
    // Float clamp → short, then short → little-endian bytes. Split into two
    // helpers so the blocking path can pitch-shift the short[] in between;
    // the streaming path uses the float→byte shortcut directly.
    private static short[] pcm16Shorts(float[] audioFloats) {
        short[] shortSamples = new short[audioFloats.length];
        for (int i = 0; i < audioFloats.length; i++) {
            float f = audioFloats[i];
            if (f > 1.0f) f = 1.0f;
            if (f < -1.0f) f = -1.0f;
            shortSamples[i] = (short) (f * 32767.0f);
        }
        return shortSamples;
    }

    private static byte[] shortsToPcm16(short[] shortSamples) {
        byte[] pcmData = new byte[shortSamples.length * 2];
        for (int i = 0; i < shortSamples.length; i++) {
            pcmData[i * 2]     = (byte) (shortSamples[i] & 0xff);
            pcmData[i * 2 + 1] = (byte) ((shortSamples[i] >> 8) & 0xff);
        }
        return pcmData;
    }

    private static byte[] floatsToPcm16(float[] audioFloats) {
        return shortsToPcm16(pcm16Shorts(audioFloats));
    }

    // ── Generate audio PCM (blocking) ────────────────────────────────────────
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

            GeneratedAudio audio = localTts.generateWithConfig(inputText.trim(), buildConfig(speedValue));

            if (cancelRequested) return null;
            if (audio == null) return null;

            float[] audioFloats = audio.getSamples();
            if (audioFloats == null || audioFloats.length == 0) return null;

            short[] shortSamples = pcm16Shorts(audioFloats);

            // Sonic pitch shifting — identical shape to the sibling
            // engines. The static `sonicQuality` knob lets storyvox dial
            // all engines uniformly from a single Settings toggle.
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

            return shortsToPcm16(shortSamples);

        } catch (Throwable t) {
            return null;
        }
    }

    // ── Streaming PCM callback ───────────────────────────────────────────────
    /**
     * Receives PCM chunks as the model produces them. Return 1 to continue
     * generating, 0 to ask the engine to stop early (sherpa-onnx
     * convention). Chunks are little-endian PCM16 mono at
     * {@link #getSampleRate()}.
     *
     * <p>Pitch shifting is NOT applied on the streaming path — Sonic's
     * pitch interpolation wants the whole utterance, and per-chunk
     * shifting introduces boundary artefacts. Callers that need pitch
     * should use the blocking {@link #generateAudioPCM}.</p>
     */
    public interface PcmCallback {
        int onAudioChunk(byte[] pcmChunk);
    }

    /**
     * Streaming synth via sherpa-onnx's
     * {@link OfflineTts#generateWithConfigAndCallback}. Each native audio
     * chunk is converted to PCM16 and handed to {@code callback}; the full
     * concatenated PCM is also returned for callers that want it (mirrors
     * the native API, which returns the complete {@link GeneratedAudio}
     * even while streaming). Returns null if not loaded / cancelled / on
     * error.
     */
    public byte[] generateAudioPCMStreaming(
            String inputText, float speedValue, float pitchValue, final PcmCallback callback) {
        if (cancelRequested) return null;
        if (inputText == null || inputText.trim().isEmpty()) return null;

        OfflineTts localTts;
        synchronized (this) {
            if (tts == null) return null;
            localTts = tts;
        }

        try {
            if (cancelRequested) return null;

            Function1<float[], Integer> sink = new Function1<float[], Integer>() {
                @Override
                public Integer invoke(float[] samples) {
                    if (cancelRequested) return 0;
                    if (samples == null || samples.length == 0) return 1;
                    if (callback != null) {
                        try {
                            return callback.onAudioChunk(floatsToPcm16(samples));
                        } catch (Throwable ignored) {
                            return 1;
                        }
                    }
                    return 1;
                }
            };

            GeneratedAudio audio = localTts.generateWithConfigAndCallback(
                    inputText.trim(), buildConfig(speedValue), sink);

            if (cancelRequested) return null;
            if (audio == null) return null;

            float[] audioFloats = audio.getSamples();
            if (audioFloats == null || audioFloats.length == 0) return null;

            return floatsToPcm16(audioFloats);

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

    public synchronized int getNumSpeakers() {
        if (tts == null) return 0;
        try {
            return tts.numSpeakers();
        } catch (Throwable ignored) {
            return 0;
        }
    }

    // ── Flow-matching step count ─────────────────────────────────────────────
    public void setNumSteps(int steps) {
        if (steps > 0) this.numSteps = steps;
    }

    public int getNumSteps() {
        return numSteps;
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
            activeModelDir = "";
        }
    }
}
