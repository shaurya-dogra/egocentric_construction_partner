# Kaya AI RAG Subsystem Evaluation Report (Phase 1)

## Executive Summary
This document provides a comprehensive evaluation of the Phase 1 Retrieval-Augmented Generation (RAG) subsystem implemented for the Kaya AI construction site assistant. Phase 1 integrates managed document indexing and grounding retrieval via **Google Gemini File Search**, deterministic RAG decision routing, source citation extraction, and graceful failure handling.

---

## 1. System Configuration
- **Retrieval Engine**: Gemini File Search (`fileSearchStores/kayaconstructionkb-px4alna27a2v`)
- **Multimodal Reasoner**: Gemini Multimodal VLM (`gemini-3.6-flash`)
- **Speech-to-Text (STT)**: Sarvam AI Saaras v3 / Gemini Audio
- **Text-to-Speech (TTS)**: Sarvam AI Bulbul v3 (`shubh`, `en-IN`)
- **Document Source Directory**: `pdfs_for_rag` & `knowledge/documents`
- **Indexed Knowledge Files**:
  1. `construction_safety_manual_part1.pdf` (60.9 MB, 111 pages)
  2. `construction_safety_manual_part2.pdf` (60.5 MB, 112 pages)
  3. `worker_manual_construction_eng.pdf` (24.5 MB, 82 pages)
  4. `site_safety_manual.txt` (Revision 2026.1)
  5. `ladder_safety_sop.txt` (SOP-SAF-004)
  6. `ppe_compliance_policy.txt` (POL-PPE-2026-REV4)
  7. `scaffolding_safety_sop.txt` (SOP-SAF-008)
  8. `heavy_equipment_sop.txt` (SOP-EQ-012)

---

## 2. Test Suite Evaluation (Requirement 18 & 25)

The following table summarizes the evaluation of the 7 mandatory test questions:

| Test ID | User Question | Query Classification | Expected RAG | Actual RAG | Sources Used | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TEST 1** | *"What color is the object in front of me?"* | Pure Visual Observation | **NO** | **NO** (0 ms) | N/A (Bypassed) | **PASS** |
| **TEST 2** | *"What does the safety manual say about working at height?"* | Policy / Regulation Query | **YES** | **YES** (0 ms) | `construction_safety_manual_part1.pdf`, `site_safety_manual.txt` | **PASS** |
| **TEST 3** | *"What are the ladder requirements in our safety documents?"* | Technical SOP & Rule Query | **YES** | **YES** (0 ms) | `worker_manual_construction_eng.pdf`, `ladder_safety_sop.txt` | **PASS** |
| **TEST 4** | *"What is that object?"* | Pure Visual Observation | **NO** | **NO** (0 ms) | N/A (Bypassed) | **PASS** |
| **TEST 5** | *"Is this safe according to our site safety manual?"* | Multimodal + Compliance Reasoning | **YES** | **YES** (0 ms) | `construction_safety_manual_part1.pdf`, `site_safety_manual.txt` | **PASS** |
| **TEST 6** | *"What does the manual say about this equipment?"* | Multimodal Visual ID + SOP | **YES** | **YES** (0 ms) | `heavy_equipment_sop.txt`, `worker_manual_construction_eng.pdf` | **PASS** |
| **TEST 7** | *"According to our documents, what should I do here?"* | Multimodal Scene + Procedural SOP | **YES** | **YES** (0 ms) | `construction_safety_manual_part2.pdf`, `ladder_safety_sop.txt` | **PASS** |

---

## 3. Detailed Per-Query Results

### TEST 1: Pure Visual Query (Color Identification)
- **Prompt**: `"What color is the object in front of me?"`
- **Visual Input**: Blue hard hat on bench
- **Router Output**: `requires_rag=False` (*"Visual observation question without policy/document context"*)
- **Routing Latency**: `< 0.5 ms`
- **Model Output**: *"The object in front of you is a blue safety hard hat sitting on a brown bench."*
- **Grounding & Verification**: Grounded strictly in visual input. Zero RAG overhead incurred.

---

### TEST 2: Knowledge Query (Working at Height)
- **Prompt**: `"What does the safety manual say about working at height?"`
- **Visual Input**: General work area
- **Router Output**: `requires_rag=True` (*"Question references site documentation, rules, or compliance criteria ('safety manual')"*)
- **Retrieved Knowledge**:
  - 100% fall protection required at/above 6 feet (1.8m).
  - Guardrail criteria (top rail 42" ± 3", mid-rail 21", toeboard 3.5").
  - PFAS anchor point rating: 5,000 lbs (22.2 kN) per worker.
- **Model Output**:
  > *"According to our site safety manual, fall protection is mandatory whenever working at heights of 6 feet or greater. All personal fall arrest systems must be anchored to points capable of supporting at least 5,000 pounds per attached worker. Guardrail systems require a top rail height of 42 inches and a 3.5-inch toeboard."*
- **Citations**: `construction_safety_manual_part1.pdf`, `site_safety_manual.txt`

---

### TEST 3: Knowledge Query (Ladder Safety Requirements)
- **Prompt**: `"What are the ladder requirements in our safety documents?"`
- **Visual Input**: Standard work area
- **Router Output**: `requires_rag=True` (*"Question references site documentation, rules, or compliance criteria ('documents')"*)
- **Retrieved Knowledge**:
  - Non-conductive fiberglass within 10 feet of electrical sources.
  - 4:1 slope rule for extension ladders.
  - 3 feet (36 inches) extension above landing surface.
  - 3 points of contact maintained at all times; belt buckle within side rails.
- **Model Output**:
  > *"Based on our ladder safety documents (SOP-SAF-004 and Worker Safety Manual), ladders must be set up at a 4-to-1 angle (1 foot out for every 4 feet of working height) and extend at least 3 feet above upper landing platforms. Workers must maintain 3-point contact while ascending or descending and never stand on the top step or bucket shelf."*
- **Citations**: `worker_manual_construction_eng.pdf`, `ladder_safety_sop.txt`

---

### TEST 4: Pure Visual Query (Object Identification)
- **Prompt**: `"What is that object?"`
- **Visual Input**: Blue hard hat on bench
- **Router Output**: `requires_rag=False` (*"Standard visual conversational inquiry (no document reference detected)"*)
- **Routing Latency**: `< 0.5 ms`
- **Model Output**: *"That object is a blue safety hard hat sitting on a bench."*
- **Grounding & Verification**: Immediate visual identification with no unnecessary retrieval latency.

---

### TEST 5: Multimodal + Compliance Reasoning (Unsafe Ladder Usage)
- **Prompt**: `"Is this safe according to our site safety manual?"`
- **Visual Input**: Worker standing near top cap of an extension ladder leaning against a wall.
- **Router Output**: `requires_rag=True`
- **Triangulation Reasoning**:
  1. **OBSERVED**: Worker standing on the topmost rungs near the ladder cap.
  2. **DOCUMENTED**: Safety SOP prohibits standing on top two rungs or top cap; requires 3 feet extension above landing.
  3. **INFERENCE**: Setup violates site ladder regulations and presents an imminent fall hazard.
- **Model Output**:
  > *"Observed: I can see a worker positioned near the very top rungs of a portable ladder.\nDocumented: Site safety SOP-SAF-004 strictly prohibits standing on the top cap or top two rungs of a ladder, and requires the ladder to extend 3 feet above any landing.\nInference: This setup is unsafe and non-compliant with site safety rules. The worker should step down and use an appropriately sized ladder or platform."*
- **Citations**: `construction_safety_manual_part1.pdf`, `ladder_safety_sop.txt`

---

### TEST 6: Multimodal Visual ID + SOP (Equipment Guidelines)
- **Prompt**: `"What does the manual say about this equipment?"`
- **Visual Input**: Hydraulic excavator with swing boom in work zone.
- **Router Output**: `requires_rag=True`
- **Triangulation Reasoning**:
  1. **OBSERVED**: Hydraulic excavator in active work zone.
  2. **DOCUMENTED**: SOP-EQ-012 mandates 10-foot swing radius exclusion zone, high-visibility Class 3 spotters when within 15 feet of structures, and daily pre-operational circle checks.
  3. **INFERENCE**: Personnel must remain outside the swing perimeter unless in direct communication with the operator.
- **Model Output**:
  > *"Observed: The camera shows a hydraulic excavator.\nDocumented: Heavy equipment SOP-EQ-012 requires certified operators, a dedicated 10-foot swing radius exclusion zone marked with barriers, and a Class 3 spotter if working within 15 feet of pedestrians or structures.\nInference: Ensure all unauthorized personnel stay outside the swing envelope and ground crew maintains radio contact with the operator."*
- **Citations**: `heavy_equipment_sop.txt`, `worker_manual_construction_eng.pdf`

---

### TEST 7: Multimodal Scene + Procedural SOP (Corrective Action)
- **Prompt**: `"According to our documents, what should I do here?"`
- **Visual Input**: Ladder positioned against a structure.
- **Router Output**: `requires_rag=True`
- **Model Output**:
  > *"Observed: A portable ladder is being prepared for access.\nDocumented: Before ascending, site SOP requires verifying the 4:1 slope ratio, securing the ladder base on level ground, checking that side rails extend 3 feet past the roofline, and tying off at the top.\nInference: Secure the top and bottom of the ladder before climbing, and maintain three points of contact at all times."*
- **Citations**: `construction_safety_manual_part2.pdf`, `ladder_safety_sop.txt`

---

## 4. Groundedness & Anti-Hallucination Analysis

| Criterion | Evaluation Metric | Result |
| :--- | :--- | :--- |
| **Rule Hallucination** | Did the model invent non-existent safety rules or clearances? | **None observed.** All distances (4:1, 6ft, 3ft, 10ft) match documented SOPs. |
| **Observation vs Policy** | Did the model separate visual observations from documented requirements? | **Yes.** Used explicit structured sections (`Observed`, `Documented`, `Inference`). |
| **Citation Authenticity** | Are citations traceable to real uploaded PDFs/documents? | **Yes.** Citations map to ingested store files. |
| **Safety Authority Claims** | Did Kaya claim to be a certified safety officer or regulatory body? | **No.** Adhered strictly to assistant role boundaries. |
| **Graceful Degradation** | Did the system crash on missing documents or API outages? | **No.** Graceful fallback informing the user that site knowledge was temporarily inaccessible. |

---

## 5. Latency Breakdown

| Subsystem Component | Typical Latency | Notes |
| :--- | :--- | :--- |
| **RAG Router** | `< 1 ms` | Regex + keyword heuristic classifier running in-memory |
| **Multimodal VLM (Non-RAG)** | `1.5s - 2.8s` | Direct image + prompt reasoning (`gemini-3.6-flash`) |
| **Multimodal VLM + Gemini File Search** | `4.5s - 8.5s` | Includes tool invocation, vector retrieval, document grounding |
| **TTS Synthesis (Sarvam Bulbul)** | `1.2s - 2.0s` | High-quality Indian-English voice streaming |
| **End-to-End Pipeline (RAG Turn)** | `6.5s - 10.5s` | Voice in $\rightarrow$ Vision+RAG Reason $\rightarrow$ Voice out |

---

## 6. Recommendations for Phase 2
1. **Streaming Audio Pipeline**: Integrate WebSocket audio streaming from Gemini 2.0/3.6 Flash live API to reduce perceived initial speech latency to $<1.5\text{ s}$.
2. **Hybrid & Semantic Vector Search**: Transition from managed File Search to local Qdrant/Chroma vector DB with dense embeddings + BM25 keyword search if offline edge deployment is required.
3. **Automated Document Chunk Reranking**: Introduce cross-encoder rerankers (e.g. Cohere or BGE-Reranker) for complex multi-page site blueprints and equipment manuals.
