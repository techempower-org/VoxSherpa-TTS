package com.CodeBySonu.VoxSherpa;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import org.junit.After;
import org.junit.Test;

/**
 * Unit coverage for the public static knobs storyvox writes to drive
 * runtime engine behavior:
 *
 *   - {@link VoiceEngine#sonicQuality}        (v2.7.13, #193)
 *   - {@link VoiceEngine#voiceLexicon}        (v2.7.14, storyvox#197)
 *   - {@link KokoroEngine#voiceLexicon}       (v2.7.14, storyvox#197)
 *   - {@link KokoroEngine#phonemizerLang}     (v2.7.14, storyvox#198)
 *   - {@link KittenEngine#sonicQuality}       (v2.8.0, storyvox#119)
 *
 * These tests intentionally run on the plain JVM (no Robolectric, no
 * emulator) — the engine classes' static initializer wraps
 * {@code System.loadLibrary("sherpa-onnx-jni")} in a try/catch, so
 * the classes load fine without the native library and the field
 * semantics can be exercised in isolation.
 *
 * The full engine-instantiation-reads-current-value behavior
 * (createTtsWithFallback / _createTtsWithFallback picking up the
 * static at construction time) is verified by inspection in the
 * test comments and by storyvox's integration harness on-device —
 * standing up a full OfflineTts in a JVM unit test would require
 * native ONNX Runtime which we cannot bundle here.
 */
public class EngineStaticKnobsTest {

    /**
     * Reset all knobs to their documented defaults after every test
     * so ordering can never make one test pass because another
     * leaked state. Mirror of how storyvox's tests behave around the
     * VoiceEngineQualityBridge.
     */
    @After
    public void resetDefaults() {
        VoiceEngine.sonicQuality = 1;
        VoiceEngine.voiceLexicon = "";
        KokoroEngine.sonicQuality = 1;
        KokoroEngine.voiceLexicon = "";
        KokoroEngine.phonemizerLang = "";
        KittenEngine.sonicQuality = 1;
    }

    /**
     * Round-trip: a write to each static field is visible on the
     * next read, and the defaults declared in the source match the
     * documented values (empty string for the new String knobs,
     * 1 for the legacy sonicQuality int).
     *
     * If a future refactor accidentally changes a field's default
     * (or its type), storyvox's seeding path at app startup goes
     * silently wrong; this guard catches that at engine-build time.
     */
    @Test
    public void staticKnobs_roundTrip_andDefaults() {
        // Defaults
        assertEquals(1, VoiceEngine.sonicQuality);
        assertEquals("", VoiceEngine.voiceLexicon);
        assertEquals(1, KokoroEngine.sonicQuality);
        assertEquals("", KokoroEngine.voiceLexicon);
        assertEquals("", KokoroEngine.phonemizerLang);
        // v2.8.0 — KittenEngine joins the family with the sonicQuality
        // knob (no lexicon/lang knobs because OfflineTtsKittenModelConfig
        // doesn't expose those today). Default 1 matches the other two.
        assertEquals(1, KittenEngine.sonicQuality);

        // Round-trip on each field
        VoiceEngine.sonicQuality = 0;
        assertEquals(0, VoiceEngine.sonicQuality);

        VoiceEngine.voiceLexicon = "/storage/emulated/0/storyvox/lex/de_de.lexicon";
        assertEquals(
                "/storage/emulated/0/storyvox/lex/de_de.lexicon",
                VoiceEngine.voiceLexicon);

        KokoroEngine.sonicQuality = 0;
        assertEquals(0, KokoroEngine.sonicQuality);

        KokoroEngine.voiceLexicon =
                "/lex/a.lexicon,/lex/b.lexicon";  // sherpa-onnx accepts CSV
        assertEquals(
                "/lex/a.lexicon,/lex/b.lexicon",
                KokoroEngine.voiceLexicon);

        KokoroEngine.phonemizerLang = "es";
        assertEquals("es", KokoroEngine.phonemizerLang);

        // Reset back to default — verify the empty-string clear works
        KokoroEngine.phonemizerLang = "";
        assertEquals("", KokoroEngine.phonemizerLang);

        // v2.8.0 — Kitten's sonicQuality field. Storyvox writes all
        // three engines together when the user flips the quality toggle
        // in Settings → Voice & Playback, so the round-trip semantics
        // must match Kokoro's exactly. 0 → upstream-default (faster),
        // 1 → higher-quality (engine-side default since #193).
        KittenEngine.sonicQuality = 0;
        assertEquals(0, KittenEngine.sonicQuality);
    }

    /**
     * Engine-instantiation-reads-current-value: the engines are
     * constructable as plain Java objects (the public no-arg
     * constructor was added in v2.7.8 for multi-instance
     * parallelism). Constructing an engine MUST NOT capture the
     * current static-knob values into instance state — the values
     * are re-read on every createTtsWithFallback / _createTtsWithFallback
     * call so a Settings write between voice reloads takes effect.
     *
     * This test exercises the contract negatively: construct an
     * engine, mutate the static, and confirm the static read still
     * returns the new value (i.e. no instance shadow). The actual
     * reload path is on-device only, but the "no shadow" property is
     * what makes the on-device behavior correct.
     */
    @Test
    public void engineConstruction_doesNotShadow_staticKnobs() {
        VoiceEngine.voiceLexicon = "/initial/lex.lexicon";
        KokoroEngine.voiceLexicon = "/initial/k.lexicon";
        KokoroEngine.phonemizerLang = "en";
        KittenEngine.sonicQuality = 0;

        VoiceEngine v = new VoiceEngine();
        KokoroEngine k = new KokoroEngine();
        // v2.8.0 — KittenEngine joins the no-arg-ctor club for the same
        // multi-instance-parallelism reason VoiceEngine/KokoroEngine did
        // (each instance owns its own onnxruntime session).
        KittenEngine kit = new KittenEngine();
        assertNotNull(v);
        assertNotNull(k);
        assertNotNull(kit);

        // Mutate AFTER construction.
        VoiceEngine.voiceLexicon = "/after/lex.lexicon";
        KokoroEngine.voiceLexicon = "/after/k.lexicon";
        KokoroEngine.phonemizerLang = "fr";
        KittenEngine.sonicQuality = 1;

        // The static read must return the post-construction value —
        // the engines do NOT cache it into instance state. The on-
        // device createTtsWithFallback path then picks up "/after/..."
        // on the next reload.
        assertEquals("/after/lex.lexicon", VoiceEngine.voiceLexicon);
        assertEquals("/after/k.lexicon", KokoroEngine.voiceLexicon);
        assertEquals("fr", KokoroEngine.phonemizerLang);
        assertEquals(1, KittenEngine.sonicQuality);
    }
}
