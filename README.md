# AgriVision Pro: Dual-Engine Potato Disease Classifier

An advanced agricultural computer vision application that evaluates edge-vs-cloud trade-offs by running a localized deep learning model alongside a cloud-based multimodal AI.

## 🛡️ Three-Tier Gatekeeper Pipeline
To optimize cloud infrastructure costs and prevent adversarial inputs, all uploads pass through a local defensive boundary:
1. **Haar Cascade Filter:** Aborts processing instantly if a human face is detected.
2. **MobileNetV2 Semantic Scanner:** Checks the top 5 ImageNet predictions to block non-agricultural objects (e.g., jerseys, vehicles, animals) with a >5% confidence threshold.
3. **HSV Organic Filter:** Verifies that the image contains at least 10% organic matter (greens, yellows, browns) before invoking downstream classification.

## 🤖 Model Comparison
* **Local Engine:** Custom Convolutional Neural Network (`potato_model_best.keras`) optimized for low-latency classification + custom HSV-based tissue lesion pixel counting for severity calculations.
* **Cloud Engine:** Google Gemini 2.5 Flash API handling advanced semantic disease indexing, symptom breakdown, and generating structured JSON treatment protocols.