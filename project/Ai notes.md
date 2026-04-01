### (Business/Research) questions (SMART)

Specific: Which pretrained vision model best classifies 12 disaster/accident image classes for this dataset?
Specific: How much does transfer learning with frozen backbones + new classifier head improve class-level recognition versus relying on a single metric?
Measurable: Compare models using weighted F1, macro F1, balanced accuracy, accuracy, and recall across folds.
Achievable: Restrict candidates to five widely used pretrained architectures and fixed training budget (epochs, batch size, optimizer, scheduler).
Relevant: Accurate disaster-type recognition supports faster triage, monitoring, and downstream decision support.
Time-bound: Complete model comparison with 5-fold CV, then retrain selected model and evaluate once on held-out test data.


### Source data

Dataset organized in 12 class folders; potential risks considered: 
Class imbalance between categories.
Mixed image properties (size, possible grayscale/RGB differences).
Potential label noise/ambiguity in visually similar disaster types.
Detection actions: 
Counted class frequencies and visualized class distribution.
Performed sample visualization per class for qualitative sanity checking.
Used stratified splitting for train/validation/test and Stratified K-Fold to preserve label proportions.
Forced image conversion to RGB before transforms to avoid channel mismatch.
Quality issues found: 
Noticeable class imbalance across categories.
Likely intra-class variability and inter-class overlap (hard classes).
Mitigations applied: 
Class-balanced loss weights via compute_class_weight.
WeightedRandomSampler for training batches.
Metric set expanded beyond plain accuracy (macro/weighted F1, balanced accuracy).
Remaining unresolved risks: 
No explicit duplicate detection.
No formal annotation audit to quantify label noise.


### Method

Pipeline steps: 
Load images with folder-based labels and inspect class balance.
Split into train+validation and hold-out test with stratification.
Apply image representation: 
Resize to 256, center-crop to 224, normalize with ImageNet mean/std.
Train-time augmentation: horizontal flip + small rotation.
Representation rationale: 
224x224 normalized tensors match pretrained backbone expectations and reduce domain shift from ImageNet pretraining.
Build and compare ML methods: 
ResNet50, VGG16, DenseNet121, MobileNetV3-Large, ViT-B/16.
Transfer-learning setup: 
Freeze feature extractor; train classification head (feature extraction mode).
Optimization setup: 
Adam (lr 0.001), cross-entropy with class weights, StepLR scheduler, batch size 32.
Validation method (explicit): 
5-fold Stratified K-Fold on train+validation portion for model comparison.
Final unbiased evaluation on untouched hold-out test split.
Comparison method: 
Per-fold tracking of train/validation loss and metrics.
Aggregate fold means/std; select robust best model using mean F1 minus stability penalty (std term).
Final model stage: 
Retrain selected model on training split with internal validation monitoring, then evaluate on test set.
Extra diagnostics: 
Confusion matrix and metric trend plots across epochs/models.


### Results

Model-comparison outputs produced: 
Cross-validated means/std for weighted F1, macro F1, balanced accuracy, accuracy, recall.
Multi-model line plot comparing key metrics.
Per-epoch validation curves (accuracy, balanced accuracy, F1) averaged over folds.
Final model outputs produced: 
Test metrics: accuracy, balanced accuracy, precision, recall, weighted F1, macro F1.
Normalized confusion matrix for class-wise error patterns.
Suggested interpretation bullets for your report: 
Higher weighted F1 indicates stronger overall performance under imbalance.
Macro F1 reveals whether minority classes are handled fairly.
Balanced accuracy confirms class-imbalance robustness.
Small fold std indicates stable generalization across splits.
Confusion matrix highlights systematic confusions between visually similar classes.
Conclusion pattern to state with your actual numbers: 
Best architecture selected by robust CV criterion outperformed alternatives on weighted F1 while maintaining competitive macro F1 and balanced accuracy.
Test performance remained aligned with CV trends, indicating limited overfitting.


### Reliability of results

Reliability actions taken: 
Stratified K-Fold instead of single split for variance-aware estimation.
Separate hold-out test set not used in model selection.
Multiple imbalance-aware metrics, not only accuracy.
Weighted sampling + weighted loss to reduce skew bias.
Limitations: 
Dataset size and class imbalance may still cause unstable minority-class estimates.
Possible label noise and no formal inter-annotator validation.
Single dataset domain; external validity across regions/sensors unknown.
Reliability claim: 
Conclusions are moderately strong for this dataset setting; strongest evidence is comparative ranking between candidate models, not universal real-world performance guarantees.


### Technical depth

Techniques beyond basic transfer learning: 
Robust model selection criterion combining mean performance and fold variability.
Dual imbalance handling at data-loader and loss-function levels.
Multi-metric evaluation framework including macro and balanced metrics.
Structured experiment logging (histories, predictions, confusion matrices) for reproducibility.
Practical computer-vision engineering: 
Custom subset wrapper with RGB coercion and transform control.
Learned from: 
PyTorch transfer-learning documentation, sklearn model-selection/metrics tools, and course methods on validation and evaluation under imbalance.


### Conclusions & recommendations

Target stakeholders: 
Emergency-response planners, risk analysts, and organizations needing rapid disaster-image categorization.
Main conclusions: 
Transfer learning is effective for multi-class disaster image classification with limited data.
Model ranking should rely on imbalance-aware cross-validated metrics, not accuracy alone.
Recommendations: 
Deploy the selected model as decision support, not as sole decision-maker.
Monitor class-wise errors continuously, especially minority/high-risk categories.
Periodically retrain with newly collected, better-balanced data.
Future research: 
External-dataset validation, fine-tuning deeper layers, stronger augmentations, calibration/uncertainty estimates, and label-quality auditing.


### Reflection

Challenges addressed: 
Imbalance handled with weighted sampling/loss and fairness-oriented metrics.
Overfitting/selection bias mitigated via stratified CV and held-out testing.
Course relevance: 
Core skills in preprocessing, validation strategy, transfer learning, and metric interpretation were directly applicable.
Additional useful skills: 
Data-centric labeling workflows, uncertainty quantification, experiment tracking platforms, and MLOps deployment monitoring.
Transparency on AI use: 
If ChatGPT was used, state it was for writing clarity, report structuring, and explanation drafting; not for fabricating results or replacing experimental work.


### ML issues (CV&IC)

Addressed issue: class imbalance across categories.
Mitigation: 
Class weights in cross-entropy.
WeightedRandomSampler for training batches.
Imbalance-aware evaluation (macro F1, balanced accuracy, recall).
Addressed issue: input heterogeneity.
Mitigation: 
Standardized resizing/cropping/normalization and RGB conversion.
Addressed issue: model-selection instability.
Mitigation: 
Stratified 5-fold CV and robustness-aware selection using mean and variability.
Remaining issue: 
Potential label noise not explicitly quantified.


### Generalisation capabilities of the method (CV&IC)

Evaluation strategy: 
Estimate generalization during model selection with stratified 5-fold CV on development data.
Confirm final generalization with untouched hold-out test set.
Evidence used: 
Fold mean + std (stability), test metrics (out-of-sample performance), confusion matrix (class-wise behavior).
Why this supports generalization claims: 
Performance consistency across folds reduces dependence on a lucky split.
Separate test set checks transfer from development to unseen examples.
Caveat: 
Generalization shown mainly within the same dataset distribution; external-domain generalization still needs dedicated validation.