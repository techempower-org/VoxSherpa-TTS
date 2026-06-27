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
import com.k2fsa.sherpa.onnx.OfflineTtsKokoroModelConfig;
import com.k2fsa.sherpa.onnx.GeneratedAudio;

public class KokoroEngine {

    static {
        try {
            System.loadLibrary("sherpa-onnx-jni");
        } catch (UnsatisfiedLinkError ignored) {}
    }

    /**
     * Sonic pitch-interpolation quality. Mirror of
     * {@link VoiceEngine#sonicQuality}; see that field's javadoc.
     * Held as a separate field because the two engines are
     * independent classes with no shared base. storyvox writes
     * both together when the user flips the Settings toggle.
     */
    public static volatile int sonicQuality = 1;

    /**
     * Per-voice lexicon override (storyvox #197). Mirror of
     * {@link VoiceEngine#voiceLexicon}; see that field's javadoc for
     * the full rationale and storyvox-side wiring contract. Passed
     * through to sherpa-onnx via
     * {@link OfflineTtsKokoroModelConfig#setLexicon}. Default empty
     * string = use the model's built-in lexicon.
     *
     * Storyvox sets this from Settings → Voice → per-voice Advanced
     * expander. Engine instantiation reads the value at construction
     * time via [createTtsWithFallback], so a Settings change requires
     * a voice reload to take effect.
     */
    public static volatile String voiceLexicon = "";

    /**
     * Kokoro phonemizer language override (storyvox #198). When set
     * to a non-empty language code (e.g. "es" for Spanish dialogue
     * inside an English book), the engine forces the phonemizer to
     * that language instead of using the voice metadata's
     * [KokoroVoiceHelper.VoiceItem.languageCode]. Passed through to
     * sherpa-onnx via {@link OfflineTtsKokoroModelConfig#setLang}.
     * Default empty string = use the voice's catalog language.
     *
     * Only meaningful on KokoroEngine — VoiceEngine (Piper) uses
     * espeak-ng's per-language voice files and does not have a
     * comparable runtime override. Storyvox writes this from
     * Settings → Voice → per-voice Advanced expander (the dropdown
     * is suppressed for Piper voices).
     *
     * Engine instantiation reads the value at construction time via
     * [createTtsWithFallback], so a Settings change requires a voice
     * reload to take effect.
     */
    public static volatile String phonemizerLang = "";

    private static volatile KokoroEngine instance;
    private OfflineTts tts;
    private String activeModelUri = "";
    // Tracks language context AND tokens/voices-bin paths + Context so
    // [setSilenceScale] (storyvox PR) can rebuild OfflineTts mid-session
    // against the same model files. Upstream keeps just activeLangCode.
    private String activeTokensUri = "";
    private String activeVoicesBinUri = "";
    private Context activeContext = null;
    private String activeLangCode = "";
    private String espeakDataPath = "";
    // Upstream Chinese-frontend support: extracted location of
    // tts_frontend.zip (lexicon-zh, Jieba dict, zh FST rules). Populated
    // by [extractTtsFrontendData]; consumed in [createTtsWithFallback]
    // when no per-voice lexicon override (storyvox #197) is set.
    private String ttsFrontendDataPath = "";
    private int activeSpeakerId = 31;
    private volatile boolean cancelRequested = false;

    /** Within-sentence silence scale — controls the engine-level pause
     *  applied around commas and mid-phrase punctuation. Default 0.2f
     *  matches the previously-hardcoded behavior; consumers can drive
     *  it from a UI slider (e.g. storyvox's punctuation-cadence
     *  control) via [setSilenceScale]. Setting it on a loaded engine
     *  rebuilds the OfflineTts so the change takes effect mid-session
     *  on the next generated sentence — mirrors VoiceEngine's
     *  [VoiceEngine.setNoiseScale] pattern. */
    public static final float DEFAULT_SILENCE_SCALE = 0.2f;
    private volatile float silenceScale = DEFAULT_SILENCE_SCALE;

    /**
     * Public constructor — added 2026-05-09 (jphein fork) to enable
     * multi-instance parallelism, mirroring [VoiceEngine]'s public
     * constructor in v2.7.8. Each instance owns its own onnxruntime
     * session via the [tts] field; calls into [generateAudioPCM] are
     * serialized per-instance via the synchronized monitor on `this`.
     * Two instances loaded against the same model files → two
     * independent OrtSessions, generate runs in parallel.
     *
     * Memory cost: each loaded Kokoro session is ~325 MB resident
     * (the multi-speaker 53-voice model). Multi-instance use should
     * verify the device has enough RAM; 8 instances on an 8 GB phone
     * fits but stresses the LMK headroom.
     *
     * Existing callers continue to use [getInstance] for the singleton
     * path; storyvox's parallel-synth slider (#88) constructs
     * additional instances as `new KokoroEngine()`.
     */
    public KokoroEngine() {}

    // ── Singleton — thread-safe double-checked locking ───────────────────────
    public static KokoroEngine getInstance() {
        if (instance == null) {
            synchronized (KokoroEngine.class) {
                if (instance == null) {
                    instance = new KokoroEngine();
                }
            }
        }
        return instance;
    }

    // ── Cancel ───────────────────────────────────────────────────────────────
    public void cancel() {
        cancelRequested = true;
    }

    // ── Smart thread count ───────────────────────────────────────────────────
    //
    // 2026-05-09 (jphein fork) — bumped the 4/6/8-core caps so sherpa-onnx
    // utilizes ALL available cores during Kokoro inference. See VoiceEngine
    // for the full rationale; mirrored here so both engines respect the
    // same multi-core policy regardless of which voice is loaded.
    //
    // v2.7.10 — [explicitNumThreads] overrides the auto heuristic when set
    // to a positive value via [loadModel]'s 5-arg overload. Lets storyvox
    // dial back from "all 8 cores" on thermally-constrained chips
    // (Snapdragon 888 famously throttles after sustained inference).
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

    // ── tts_frontend extract (Chinese lexicon + Jieba dictionaries) ──────────
    // Upstream "Chinese Voice crash fix" (CodeBySonu95). Unpacks
    // tts_frontend.zip into files/tts_frontend_data on first use; the
    // extracted lexicons + Jieba dict + FST rules let Kokoro render
    // Chinese without crashing eSpeak on out-of-vocabulary words.
    private synchronized void extractTtsFrontendData(Context context) {
        if (context == null) return;
        File destDir = new File(context.getFilesDir(), "tts_frontend_data");
        String[] existing = destDir.list();

        if (!destDir.exists() || existing == null || existing.length == 0) {
            destDir.mkdirs();
            try (InputStream is = context.getAssets().open("tts_frontend.zip");
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
        ttsFrontendDataPath = destDir.getAbsolutePath();
    }

    // ── Provider fallback: XNNPACK → CPU ────────────────────────────────────
    private OfflineTts createTtsWithFallback(String onnxPath, String tokensPath, String voicesBinPath) {
        String[] providers = {"xnnpack", "cpu"};

        for (String provider : providers) {
            try {
                // Cancel check before initialization
                if (cancelRequested) return null;

                KokoroVoiceHelper.VoiceItem currentVoice = KokoroVoiceHelper.getById(activeSpeakerId);
                String langCode = (currentVoice != null) ? currentVoice.languageCode : "en";

                OfflineTtsKokoroModelConfig kokoroConfig = new OfflineTtsKokoroModelConfig();
                kokoroConfig.setModel(onnxPath);
                kokoroConfig.setTokens(tokensPath);
                kokoroConfig.setVoices(voicesBinPath);
                kokoroConfig.setDataDir(espeakDataPath);

                // ── Phonemizer language ──────────────────────────────────
                // storyvox #198 — an explicit phonemizer-language override
                // wins (e.g. "es" for Spanish dialogue inside an English
                // book). Otherwise fall back to upstream's per-speaker
                // behavior: Chinese (zh) is remapped to "en-us" to prevent
                // an eSpeak crash on out-of-vocabulary words; every other
                // language passes its native langCode so eSpeak-ng loads the
                // right dictionary. Read at construction time so Settings
                // updates take effect on the next reload.
                String overrideLang = phonemizerLang;
                String effectiveLang;
                if (overrideLang != null && !overrideLang.isEmpty()) {
                    effectiveLang = overrideLang;
                } else if ("zh".equalsIgnoreCase(langCode)) {
                    effectiveLang = "en-us";
                } else {
                    effectiveLang = langCode;
                }
                kokoroConfig.setLang(effectiveLang);

                OfflineTtsConfig config = new OfflineTtsConfig();

                // ── Lexicon / Chinese frontend ───────────────────────────
                // storyvox #197 — an explicit per-voice lexicon override
                // replaces the default lexicon wiring (empty string = no
                // override). When unset, fall back to upstream's
                // multi-lingual setup: English + Chinese lexicons, the Jieba
                // segmentation dictionary, and the Chinese FST rules
                // (phone/date/number) extracted from tts_frontend.zip.
                String lex = voiceLexicon;
                if (lex != null && !lex.isEmpty()) {
                    kokoroConfig.setLexicon(lex);
                } else if (ttsFrontendDataPath != null && !ttsFrontendDataPath.isEmpty()) {
                    String lexiconEn = new File(ttsFrontendDataPath, "lexicon-us-en.txt").getAbsolutePath();
                    String lexiconZh = new File(ttsFrontendDataPath, "lexicon-zh.txt").getAbsolutePath();

                    // Always set both lexicons for multi-language support
                    kokoroConfig.setLexicon(lexiconEn + "," + lexiconZh);

                    // Set Jieba dictionary directory for Chinese word segmentation
                    File dictFolder = new File(ttsFrontendDataPath, "dict");
                    if (dictFolder.exists()) {
                        kokoroConfig.setDictDir(dictFolder.getAbsolutePath());
                    }

                    // Set FST rules for Chinese parsing
                    String phoneZh = new File(ttsFrontendDataPath, "phone-zh.fst").getAbsolutePath();
                    String dateZh = new File(ttsFrontendDataPath, "date-zh.fst").getAbsolutePath();
                    String numberZh = new File(ttsFrontendDataPath, "number-zh.fst").getAbsolutePath();

                    config.setRuleFsts(phoneZh + "," + dateZh + "," + numberZh);
                }

                OfflineTtsModelConfig modelConfig = new OfflineTtsModelConfig();
                modelConfig.setKokoro(kokoroConfig);
                modelConfig.setNumThreads(effectiveNumThreads());
                modelConfig.setProvider(provider);
                modelConfig.setDebug(false);

                config.setModel(modelConfig);
                // jphein fork: keep maxNumSentences=3 (storyvox batches
                // sentences with internal periods like "Mr. Smith ran.";
                // upstream chose =1 for System TTS responsiveness which
                // doesn't apply here). silenceScale uses the field
                // exposed by the storyvox setter PR.
                config.setMaxNumSentences(3);
                config.setSilenceScale(silenceScale);

                OfflineTts candidate = new OfflineTts(null, config);

                // As confirmed by the user, Kokoro supports punctuations perfectly.
                GeneratedAudio test = candidate.generate("...", activeSpeakerId, 1.0f);
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
     * v2.7.10 — overload that lets the caller override the numThreads
     * passed to sherpa-onnx. Pass [numThreads] = 0 for the auto
     * heuristic (same as the 4-arg [loadModel]). Used by storyvox's
     * "Threads per engine" Settings slider so users on thermally-
     * constrained devices can dial back from the auto value.
     */
    public synchronized String loadModel(Context context, String onnxPath,
                                          String tokensPath, String voicesBinPath,
                                          int numThreads) {
        this.explicitNumThreads = numThreads;
        return loadModel(context, onnxPath, tokensPath, voicesBinPath);
    }

    public synchronized String loadModel(Context context, String onnxPath,
                                          String tokensPath, String voicesBinPath) {
        cancelRequested = false; // Reset on new load

        KokoroVoiceHelper.VoiceItem currentVoice = KokoroVoiceHelper.getById(activeSpeakerId);
        String voiceLangCode = (currentVoice != null) ? currentVoice.languageCode : "en";
        // storyvox #198 — phonemizer override participates in the
        // "already loaded?" cache key so flipping the override between
        // chapters triggers a reload instead of silently keeping the
        // old language.
        String overrideLang = phonemizerLang;
        String targetLangCode =
                (overrideLang != null && !overrideLang.isEmpty())
                        ? overrideLang
                        : voiceLangCode;

        // Avoid reloading if the exact same model and language code are already active
        if (tts != null && activeModelUri.equals(onnxPath) && activeLangCode.equals(targetLangCode)) {
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
            extractTtsFrontendData(context); // upstream: Chinese lexicon + dictionaries

            if (espeakDataPath == null || espeakDataPath.isEmpty())
                return "Error: espeak-ng-data extraction failed.";

            tts = createTtsWithFallback(onnxPath, tokensPath, voicesBinPath);

            if (tts == null) return "Error: Model load failed on all providers.";

            activeModelUri = onnxPath;
            activeTokensUri = tokensPath;
            activeVoicesBinUri = voicesBinPath;
            activeContext = context.getApplicationContext();
            activeLangCode = targetLangCode;
            return "Success";

        } catch (Throwable t) {
            activeModelUri = "";
            activeLangCode = "";
            tts = null;
            return "Error: " + (t.getMessage() != null ? t.getMessage() : t.getClass().getSimpleName());
        }
    }

    // ── Generate audio PCM ───────────────────────────────────────────────────
    public byte[] generateAudioPCM(String inputText, float speedValue, float pitchValue) {
        // Immediate cancel check
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

            // Float to Short conversion with anti-clipping bounds
            short[] shortSamples = new short[audioFloats.length];
            for (int i = 0; i < audioFloats.length; i++) {
                float f = audioFloats[i];
                if (f > 1.0f) f = 1.0f;
                if (f < -1.0f) f = -1.0f;
                shortSamples[i] = (short) (f * 32767.0f);
            }

            // Sonic pitch shifting
            if (pitchValue != 1.0f) {
                if (cancelRequested) return null;
                int sampleRate = localTts.sampleRate();
                if (sampleRate > 0) {
                    try {
                        com.CodeBySonu.VoxSherpa.Sonic sonic = new com.CodeBySonu.VoxSherpa.Sonic(sampleRate, 1);
                        // storyvox #193 — Sonic quality parameterized via
                        // KokoroEngine.sonicQuality (default 1). See the
                        // field's javadoc for rationale.
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
                        // Fallback to original samples if Sonic fails
                    }
                }
            }

            if (cancelRequested) return null;

            // Short to PCM byte array (Little Endian format required by AudioTrack)
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

    public String getActiveVoiceName() {
        try {
            KokoroVoiceHelper.VoiceItem voice = KokoroVoiceHelper.getById(activeSpeakerId);
            return voice != null ? voice.getFullLabel() : "Unknown Voice";
        } catch (Throwable ignored) {
            return "Unknown Voice";
        }
    }

    public int getNumSpeakers() {
        try {
            return KokoroVoiceHelper.getAllVoices().size();
        } catch (Throwable ignored) {
            return 0;
        }
    }

    // ── Silence scale (within-sentence pauses) ──────────────────────────────
    // The OfflineTtsConfig.silenceScale parameter controls the engine's
    // pause around commas and mid-phrase punctuation. Surfacing it as a
    // public knob lets consumers (e.g. storyvox's punctuation-cadence
    // slider) drive the cadence without forking the engine.
    //
    // Setting it on a loaded engine triggers an immediate rebuild of
    // the OfflineTts so the next [generateAudioPCM] call honors the new
    // scale — mirrors VoiceEngine's setNoiseScale + _reloadIfActive
    // pattern. The no-op fast-path skips both the assignment and the
    // reload when the value matches the active scale, so settings UIs
    // can call freely on every recomposition.
    public synchronized void setSilenceScale(float scale) {
        if (this.silenceScale == scale) return;
        this.silenceScale = scale;
        _reloadIfActive();
    }

    public float getSilenceScale() {
        return silenceScale;
    }

    // Internal: rebuild OfflineTts with the current (model, tokens,
    // voicesBin, silenceScale) values. Caller must hold the monitor on
    // `this`. No-op if no model is loaded. Mirrors the same pattern
    // VoiceEngine uses for its noise-scale setters.
    private void _reloadIfActive() {
        if (tts == null
                || activeModelUri.isEmpty()
                || activeTokensUri.isEmpty()
                || activeVoicesBinUri.isEmpty()
                || activeContext == null) {
            return;
        }
        cancelRequested = true;
        OfflineTts old = tts;
        OfflineTts replacement = createTtsWithFallback(
                activeModelUri, activeTokensUri, activeVoicesBinUri);
        if (replacement != null) {
            tts = replacement;
            try { old.release(); } catch (Throwable ignored) {}
        }
        // If replacement fails, we keep the old engine — better to
        // play with the previous silence scale than to drop into a
        // null state mid-session. cancelRequested gets reset on the
        // next loadModel call.
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
            activeLangCode = "";
        }
    }
}
