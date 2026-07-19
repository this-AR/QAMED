"""
QAMed - LLMLingua Context Compression Test

Run this standalone script to evaluate whether LLMLingua is worth integrating
into the QAMed retrieval pipeline.

Tests:
  1. Compression ratio (how much shorter is the compressed context?)
  2. Token count before vs after compression
  3. Qualitative check — does compressed text still contain key medical facts?
  4. Timing — how much latency does the compression step add?

Usage:
    python evaluation/test_llmlingua.py
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Sample medical context chunks (simulate what the pipeline retrieves)
TEST_CASES = [
    {
        "query": "What is the inguinal canal and what are its contents?",
        "chunks": [
            """The inguinal canal is a narrow oblique passage in the lower part of the anterior abdominal wall.
It extends from the deep inguinal ring (an opening in the fascia transversalis) to the superficial inguinal ring
(a triangular aperture in the external oblique aponeurosis). The canal is approximately 4 cm long in adults
and runs obliquely downward and medially. It transmits the spermatic cord in males and the round ligament of
the uterus in females, as well as the ilioinguinal nerve in both sexes. The canal is bounded anteriorly by the
external oblique aponeurosis, posteriorly by the fascia transversalis, superiorly by the arched fibers of the
internal oblique and transversus abdominis muscles, and inferiorly by the inguinal ligament.""",

            """The spermatic cord consists of the following structures:
1. Ductus deferens
2. Three arteries: testicular artery, cremasteric artery, and artery to vas deferens
3. Pampiniform venous plexus
4. Genital branch of genitofemoral nerve
5. Lymphatics draining the testis
6. Autonomic nerve fibers
7. Remains of the processus vaginalis
The cord is invested by three fascial layers: the internal spermatic fascia, cremasteric fascia,
and external spermatic fascia. These layers correspond to the three layers of the abdominal wall
through which the testis descends during fetal development.""",

            """The inguinal region is of great clinical importance. It is a site of weakness in the abdominal wall
and is frequently the site of hernia formation. Inguinal hernias are the most common type of hernia.
They can be classified as direct or indirect. Indirect inguinal hernias pass through the deep inguinal ring
lateral to the inferior epigastric vessels, while direct inguinal hernias pass through the posterior wall
of the inguinal canal, medial to the epigastric vessels, through an area called Hesselbach's triangle.
Inguinal hernias are more common in males due to the oblique course of the spermatic cord through the canal.
The processus vaginalis, a peritoneal diverticulum, may remain patent and predispose to indirect hernias."""
        ]
    },
    {
        "query": "Why are the renal arteries considered end arteries?",
        "chunks": [
            """The renal arteries are functional end arteries. Although there are some anastomoses between
intrarenal branches, these are insufficient to maintain adequate blood flow to all parts of the kidney
if any major branch is occluded. This is in contrast to true anatomical end arteries, which have no
anastomoses at all. The practical consequence of this is that occlusion of any renal artery branch
results in infarction of the corresponding renal tissue. This is why renal artery stenosis or thrombosis
can cause renal infarction and permanent loss of nephrons in the affected territory.""",

            """The blood supply of the kidney is from the renal artery, which arises from the aorta at the
level of L1-L2 vertebrae. The right renal artery is longer and passes posterior to the inferior vena cava.
Each renal artery divides into an anterior and posterior division before entering the kidney at the hilum.
These divisions give rise to segmental arteries, which further divide into interlobar arteries, arcuate
arteries, interlobular arteries, and finally afferent glomerular arterioles. Unlike the general circulation,
the renal vasculature has a portal-like arrangement with two capillary beds: glomerular capillaries
and peritubular capillaries."""
        ]
    }
]


def run_test():
    print("=" * 70)
    print("QAMed - LLMLingua Context Compression Test")
    print("=" * 70)

    try:
        from llmlingua import PromptCompressor
        print("\n[OK] LLMLingua imported successfully.\n")
    except ImportError as e:
        print(f"\n[FAIL] LLMLingua import failed: {e}")
        print("   Run: pip install llmlingua")
        return

    print("Loading PromptCompressor (llmlingua-2 mini)...")
    t0 = time.time()
    try:
        compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu"
        )
    except Exception as e:
        print(f"\n[FAIL] Failed to initialize PromptCompressor: {e}")
        return

    load_time = time.time() - t0
    print(f"   Loaded in {load_time:.1f}s\n")

    overall_ratios = []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"{'─'*70}")
        q = case['query']
        print(f"Test {i}: {q}")
        print("-" * 70)

        full_context = "\n\n".join(case["chunks"])
        original_tokens = len(full_context.split())
        print(f"Original context: {len(full_context)} chars / ~{original_tokens} tokens")

        t0 = time.time()
        try:
            compressed = compressor.compress_prompt(
                context=case["chunks"],
                instruction=case["query"],
                question=case["query"],
                target_token=original_tokens // 2,   # target 50% compression
                rank_method="longllmlingua",
            )
            compress_time = time.time() - t0
            compressed_text = compressed.get("compressed_prompt", "")
            compressed_tokens = len(compressed_text.split())
            ratio = round(compressed_tokens / max(original_tokens, 1), 3)
            overall_ratios.append(ratio)

            print(f"Compressed:        {len(compressed_text)} chars / ~{compressed_tokens} tokens")
            print(f"Compression ratio: {ratio:.1%} of original")
            print(f"Compression time:  {compress_time:.2f}s")
            print("\nCompressed text preview (first 600 chars):")
            print("." * 50)
            print(compressed_text[:600])
            print("." * 50 + "\n")

        except Exception as e:
            print(f"[FAIL] Compression failed: {e}\n")
            continue

    if overall_ratios:
        avg_ratio = sum(overall_ratios) / len(overall_ratios)
        print(f"{'='*70}")
        print(f"SUMMARY")
        print(f"  Average compression ratio: {avg_ratio:.1%} of original size")
        print(f"  Token savings: ~{(1 - avg_ratio) * 100:.0f}% fewer tokens sent to LLM")
        if avg_ratio < 0.65:
            print(f"\n[INTEGRATE] LLMLingua provides meaningful compression ({(1-avg_ratio)*100:.0f}% token savings).")
            print("   Integration is worthwhile — add after parent expansion in the pipeline.")
        else:
            print(f"\n[SKIP] Compression ratio is modest ({(1-avg_ratio)*100:.0f}% token savings).")
            print("   Consider skipping LLMLingua — the overhead may not justify the gain.")
        print("=" * 70)


if __name__ == "__main__":
    run_test()
