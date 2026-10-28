# SafeSight AI
## Complete Project Implementation Plan
### Deep Learning-Based Workplace Safety Monitoring System

**Project type:** Final-year B.Tech Computer Science Engineering project  
**Domain:** Deep Learning, Computer Vision, Industrial AI, Workplace Safety  
**Primary objective:** Develop a working application that analyzes workplace images and videos to identify potential personal protective equipment (PPE) violations and help safety officers review incidents efficiently.

---

# 1. PROJECT OVERVIEW

## 1.1 Problem Statement

Factories, construction sites, warehouses, and other industrial workplaces require workers to follow safety procedures. These include wearing helmets and high-visibility vests and avoiding restricted areas.

Manual CCTV monitoring requires continuous human attention. Safety officers may overlook violations, and reviewing recorded footage can be time-consuming.

SafeSight AI aims to automate part of this process using Deep Learning and Computer Vision. The application will analyze uploaded images or videos, detect workers and visible safety equipment, flag supported potential violations, and present the results through an interactive dashboard.

## 1.2 Proposed Solution

Build a web application that allows a user to upload a workplace image or video and receive automated safety-analysis results.

The system should:

1. Detect workers and relevant safety equipment in images.
2. Identify supported PPE violations, such as a missing helmet or safety vest, when the available model and data permit reliable assessment.
3. Analyze video frames and associate detections with timestamps.
4. Apply configurable safety rules to flag potential violations.
5. Save incident records and evidence frames for review.
6. Display incident statistics and model performance through a dashboard.
7. Allow users to review, confirm, or dismiss flagged events.
8. Provide downloadable incident reports.

The system is an assistive monitoring tool. It must not claim to guarantee workplace safety or replace a qualified safety officer.

## 1.3 Target Users

- Factory safety officers
- Construction site supervisors
- Warehouse managers
- Industrial safety teams
- Researchers working on computer vision and workplace safety

## 1.4 Project Scope

### Mandatory features — MVP

- Image upload and analysis
- Video upload and frame processing
- Worker and PPE detection
- Supported PPE violation identification
- Confidence scores and annotated images
- Timestamped incident logging
- Interactive dashboard
- CSV incident export
- Model evaluation and documentation

### Advanced features — if time permits

- Webcam-based monitoring
- Person tracking across consecutive frames
- Configurable restricted-zone detection
- Alert notifications
- LSTM-based temporal analysis for a specifically defined event
- Docker-based deployment

Do not implement all advanced features before the core detection pipeline works.

---

# 2. TECHNOLOGY STACK

Use the following stack unless there is a strong technical reason to change it.

| Component | Technology | Purpose |
|---|---|---|
| Programming language | Python 3.11 | Main development language |
| Deep Learning | PyTorch | Model training and inference |
| Object detection | Ultralytics YOLO | Worker and PPE detection |
| Image processing | OpenCV | Video reading, frame extraction and annotation |
| Numerical processing | NumPy | Array operations |
| Data analysis | Pandas | Incident analysis and evaluation |
| Web interface | Streamlit | Interactive application |
| Database | SQLite | Persistent incident records |
| Charts | Plotly | Interactive analytics |
| Testing | pytest | Unit and integration tests |
| Version control | Git and GitHub | Source code management |
| Deployment | Streamlit Community Cloud or another suitable host | Application access |
| Optional packaging | Docker | Reproducible execution |

Use a compatible set of package versions and pin them in `requirements.txt`. Check compatibility before installing or upgrading packages.

For GPU training, use a compatible CUDA-enabled PyTorch environment where available. CPU inference should remain supported for smaller demonstrations.

---

# 3. DATASET AND DATA PREPARATION

## 3.1 Primary Dataset

Use the publicly documented Construction-PPE object-detection dataset as the first candidate.

Dataset documentation:
https://docs.ultralytics.com/datasets/detect/construction-ppe/

Alternative:
https://public.roboflow.com/object-detection/hard-hat-workers/

Before training, inspect the selected dataset and verify:

- Actual class names
- Number of images and annotations
- Training, validation, and test splits
- Annotation format
- Image quality and diversity
- License and permitted usage
- Whether the dataset includes the classes required by the application

Never assume that every dataset includes both positive and negative PPE examples.

## 3.2 Dataset Requirements

The desired categories may include:

- Person or worker
- Helmet
- Head
- Safety vest
- Other available PPE categories

Use the actual labels provided by the selected dataset. Do not invent missing categories.

If a dataset lacks sufficient examples of workers without helmets or vests, obtain a suitable additional labeled dataset or narrow the project's supported claims.

## 3.3 Dataset Preprocessing

Implement a preprocessing workflow that:

1. Downloads or loads the dataset using its documented method.
2. Validates image and annotation paths.
3. Checks for corrupt or unreadable images.
4. Confirms that annotation coordinates are valid.
5. Verifies that class IDs map to the correct class names.
6. Preserves the provided train, validation, and test splits where possible.
7. Resizes images using the selected model's supported preprocessing pipeline.
8. Applies reasonable augmentations during training, such as horizontal flips, brightness changes and scaling.
9. Avoids augmentation that would distort the meaning of safety equipment.
10. Records the dataset version, source, and preprocessing settings.

Avoid data leakage. Images or near-duplicate frames from the same source video should not appear across training and test splits.

## 3.4 Dataset Documentation

Create `docs/dataset_card.md` documenting:

- Dataset name and source
- License and usage conditions
- Classes and annotation format
- Split sizes
- Known class imbalance
- Known limitations
- Preprocessing and augmentation
- How to download the data

Do not commit large datasets or restricted files to GitHub. Provide reproducible download instructions instead.

---

# 4. DEEP LEARNING APPROACH

## 4.1 Model Selection

Use a pretrained, lightweight YOLO object-detection model as the initial baseline. Fine-tune it on the selected workplace PPE dataset.

The model must identify objects and their locations, not merely assign one label to an entire image.

Start with a small model variant that is practical for the available hardware. Keep the model architecture and pretrained checkpoint configurable.

## 4.2 Why Object Detection?

An image classifier might tell us that a picture contains a worker, but an object detector can identify multiple objects and their locations within the same image.

For SafeSight AI, the system needs to locate workers and visible safety equipment. These locations can then be used by a separate rule engine to flag potential violations.

## 4.3 Training Pipeline

Implement the following workflow:

1. Load the configured dataset.
2. Initialize the pretrained detection model.
3. Verify the dataset configuration and class mapping.
4. Fine-tune the model on training data.
5. Evaluate performance on validation data.
6. Save the best-performing checkpoint.
7. Evaluate the final checkpoint on held-out test data.
8. Record the metrics and training configuration.
9. Compare the trained model against the baseline.
10. Export the model if required for deployment.

Create a configurable training script supporting:

- Image size
- Batch size
- Number of epochs
- Learning rate or optimizer settings
- Pretrained checkpoint
- Dataset configuration path
- Output directory
- Random seed where supported
- Device selection

Use early stopping or checkpoint selection where appropriate. Never report training metrics as test metrics.

## 4.4 Training and Evaluation Metrics

Record at least:

- Precision
- Recall
- F1-score
- Mean Average Precision at IoU 0.50 (mAP50)
- mAP50–95 where supported
- Per-class performance
- Confusion matrix where applicable
- Inference latency
- Model size

Do not hardcode expected accuracy. Display only values obtained from actual experiments.

## 4.5 PPE Violation Logic

The object detector and the safety rule engine must be separate modules.

For example, detecting a person and a helmet in the same image does not automatically prove that the person is wearing the helmet.

Implement a documented association method that uses relevant detections and spatial relationships. If the dataset includes head detections, use them where appropriate.

For a missing-helmet rule:

1. Detect the person or worker.
2. Identify the associated head region when supported.
3. Check for an associated helmet detection.
4. Apply configurable confidence and association thresholds.
5. Flag a potential missing-helmet event only when the rule's conditions are satisfied.
6. Mark uncertain cases for human review.

Implement the same approach for safety vests only if the available annotations and model support it.

Provide an `uncertain` or `needs_review` outcome where evidence is insufficient. Do not interpret an undetected helmet as conclusive proof that no helmet exists.

## 4.6 Restricted-Zone Detection

Treat restricted-zone monitoring as an optional feature.

Allow the user to define a polygonal zone on a frame. Track detected people where feasible and check whether their positions enter that polygon.

Document the limitations of simple point-in-polygon rules. A bounding-box center inside a zone is only a proxy for entry, and camera calibration may be necessary for more reliable industrial use.

---

# 5. SYSTEM ARCHITECTURE

Implement the application using modular components.

### Component 1: User Interface

Accept image and video uploads, expose configuration settings, and display analysis results.

### Component 2: Media Processor

Validate uploaded files, decode images and videos, extract frames, and preserve frame timestamps.

### Component 3: Detection Engine

Load the trained model, run inference, and return object labels, bounding boxes and confidence scores.

### Component 4: Safety Rule Engine

Associate workers with detected PPE and apply configured safety rules.

### Component 5: Incident Manager

Create incident records, save evidence frames, and support review, confirmation and dismissal.

### Component 6: Database

Persist incident metadata and review status using SQLite.

### Component 7: Analytics Engine

Calculate event counts, category breakdowns, review outcomes and trends.

### Component 8: Reporting

Export filtered incident records to CSV and optionally generate a printable summary.

### Component 9: Evaluation Module

Calculate model metrics using labeled validation or test data and produce evaluation artifacts.

---

# 6. RECOMMENDED PROJECT STRUCTURE

Create a clean, maintainable repository with the following layout:

```text
SafeSight-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
│
├── config/
│   └── settings.yaml
│
├── src/
│   ├── data/
│   │   ├── dataset_validation.py
│   │   └── preprocessing.py
│   │
│   ├── models/
│   │   ├── detector.py
│   │   └── model_loader.py
│   │
│   ├── video/
│   │   ├── video_processor.py
│   │   └── tracking.py
│   │
│   ├── safety/
│   │   ├── ppe_rules.py
│   │   └── zone_rules.py
│   │
│   ├── incidents/
│   │   ├── incident_manager.py
│   │   └── database.py
│   │
│   ├── analytics/
│   │   └── metrics.py
│   │
│   └── utils/
│       ├── file_validation.py
│       └── logging_config.py
│
├── pages/
│   ├── 1_Image_Analysis.py
│   ├── 2_Video_Analysis.py
│   ├── 3_Incident_Management.py
│   ├── 4_Analytics.py
│   └── 5_Model_Evaluation.py
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── tests/
│   ├── test_file_validation.py
│   ├── test_ppe_rules.py
│   ├── test_incidents.py
│   └── test_video_processor.py
│
├── docs/
│   ├── architecture.md
│   ├── dataset_card.md
│   ├── model_card.md
│   └── evaluation.md
│
├── models/
│   └── .gitkeep
│
├── data/
│   └── .gitkeep
│
└── outputs/
    └── .gitkeep
```

Adapt the structure if needed, but maintain clear separation between model inference, safety logic, storage and UI.

Do not commit trained weights, uploaded videos, sensitive footage, or large datasets unless explicitly permitted and necessary.

---

# 7. APPLICATION USER INTERFACE

Build a professional industrial analytics dashboard with a consistent design system.

## 7.1 Visual Design

Use:

- A dark navy or charcoal sidebar
- A clean light main workspace
- A restrained orange or amber accent for warnings
- Green for reviewed or clear statuses
- Red for confirmed critical incidents
- Accessible contrast and readable typography
- Responsive layouts for laptop and desktop screens

Do not use decorative graphics that reduce the space available for useful information.

## 7.2 Sidebar Navigation

Include:

- Dashboard
- Image Analysis
- Video Analysis
- Incident Management
- Analytics
- Model Evaluation
- Settings

## 7.3 Dashboard Page

Display:

- Total videos or images analyzed
- Total flagged events
- Events awaiting review
- Confirmed incidents
- PPE violation breakdown
- Event trends over time
- Recent incident records

Use real database values. Show an informative empty state when no analysis has been run.

## 7.4 Image Analysis Page

Features:

1. Upload JPG, JPEG or PNG.
2. Preview the original image.
3. Run detection.
4. Display the annotated image.
5. Show detected classes and confidence scores.
6. Display any potential PPE violations.
7. Allow the user to save the analysis to the incident log.
8. Provide a download option for the annotated image.

## 7.5 Video Analysis Page

Features:

1. Upload a supported video file.
2. Validate file type, size and readability.
3. Display the original video.
4. Configure inference settings where appropriate.
5. Process frames at a configurable sampling rate.
6. Preserve timestamps corresponding to the original video.
7. Display an annotated preview.
8. Create incident records for supported potential violations.
9. Show processing progress and errors.
10. Allow the user to download the processed preview or incident report.

Avoid loading every frame into memory at once. Process the video incrementally.

## 7.6 Incident Management Page

Display a searchable and filterable table containing:

- Incident ID
- Timestamp
- Source filename
- Violation category
- Confidence or rule score
- Evidence image
- Review status
- Reviewer notes

Allow reviewers to:

- Open the evidence frame
- Confirm an event
- Dismiss a false positive
- Mark an event as uncertain
- Add a note
- Filter records by date, category and status
- Export selected records

Keep model predictions and human review decisions in separate fields.

## 7.7 Analytics Page

Include:

- Bar chart of events by category
- Line chart of event counts over time
- Review-status breakdown
- Source-wise incident distribution
- Filters for date and event type

Label charts clearly. Distinguish detected events from confirmed incidents.

## 7.8 Model Evaluation Page

Display:

- Model name and checkpoint information
- Dataset and evaluation split
- Precision, recall, F1-score and mAP where applicable
- Per-class results
- Confusion matrix where supported
- Inference latency
- Known limitations

The UI must distinguish metrics from an offline labeled evaluation dataset from events detected in user-uploaded media.

---

# 8. DATABASE DESIGN

Use SQLite for persistent storage.

Create an `incidents` table containing:

- `id`: primary key
- `source_filename`: uploaded media name
- `source_type`: image or video
- `timestamp_seconds`: video-relative timestamp, if applicable
- `event_category`: potential violation category
- `confidence`: model confidence where relevant
- `rule_status`: result of the safety rule
- `review_status`: pending, confirmed, dismissed or uncertain
- `evidence_path`: path to the saved evidence image
- `reviewer_notes`: optional notes
- `created_at`: record creation timestamp

Create additional tables only when necessary, such as a separate analysis-session table.

Requirements:

- Use parameterized SQL queries.
- Persist records across application restarts.
- Handle duplicate submissions sensibly.
- Validate all paths and user inputs.
- Provide a clear method to remove test data.
- Never expose internal filesystem paths unnecessarily.

Do not store sensitive video footage indefinitely. Make retention behavior explicit and configurable.

---

# 9. IMPLEMENTATION PHASES

Implement the project in the following order.

## Phase 1: Environment and Repository Setup

- Create the repository.
- Create and activate a Python virtual environment.
- Install compatible dependencies.
- Create the initial folder structure.
- Add configuration and logging.
- Add a basic Streamlit page.
- Verify that the application launches successfully.

**Acceptance criteria:** The application starts without errors and displays the initial dashboard.

## Phase 2: Dataset Preparation

- Download the chosen public dataset.
- Verify its license and class labels.
- Validate annotations and image paths.
- Inspect sample images and bounding boxes.
- Confirm the train, validation and test splits.
- Create the dataset card.

**Acceptance criteria:** A validation script reports dataset statistics and identifies invalid or missing annotations.

## Phase 3: Baseline Model

- Load a pretrained lightweight detector.
- Test inference on sample images.
- Confirm that detected labels match the model configuration.
- Save annotated predictions.
- Record baseline metrics if labeled evaluation data is available.

**Acceptance criteria:** The baseline produces valid detections and the inference script runs independently of the UI.

## Phase 4: Model Fine-Tuning

- Fine-tune the model using the selected dataset.
- Monitor training and validation performance.
- Save the best checkpoint.
- Evaluate on held-out data.
- Record metrics and training settings.
- Compare baseline and fine-tuned results.

**Acceptance criteria:** The trained checkpoint can be loaded independently, and its evaluation results are reproducible.

## Phase 5: PPE Rule Engine

- Implement worker-to-PPE association.
- Implement supported missing-helmet logic.
- Implement supported missing-vest logic.
- Add confidence thresholds.
- Handle uncertain detections.
- Write unit tests for safety rules.

**Acceptance criteria:** Rule tests cover valid detections, missing PPE, unrelated PPE detections, low confidence and ambiguous cases.

## Phase 6: Image Analysis

- Implement file upload and validation.
- Connect the detector.
- Display annotated results.
- Show detection details.
- Save selected results to the database.

**Acceptance criteria:** A user can upload an image, review its detections and save an incident record.

## Phase 7: Video Analysis

- Implement incremental frame reading.
- Preserve timestamps.
- Run inference on sampled frames.
- Apply the safety rule engine.
- Save evidence frames and event records.
- Display progress and handle decoding errors.

**Acceptance criteria:** A short test video can be analyzed without loading the entire video into memory.

## Phase 8: Incident Management and Analytics

- Build incident filters and review actions.
- Add summary metrics and charts.
- Implement CSV export.
- Add empty states and error messages.

**Acceptance criteria:** Saved records remain available after restarting the application, and dashboard values reflect actual records.

## Phase 9: Testing and Optimization

- Run unit and integration tests.
- Test corrupt files and unsupported formats.
- Test low-confidence detections.
- Measure inference speed.
- Check memory usage on longer videos.
- Verify that review decisions persist.
- Resolve major errors and document limitations.

**Acceptance criteria:** Core workflows pass the defined tests and failures produce clear messages instead of application crashes.

## Phase 10: Documentation and Deployment

- Complete the README.
- Document setup and training.
- Create the architecture diagram.
- Add a model card and evaluation report.
- Record a demonstration video.
- Deploy to a suitable environment.
- Test the deployed application with sample media.

**Acceptance criteria:** Another developer can reproduce the environment, load the model and run the application using the documented instructions.

---

# 10. TESTING PLAN

Create tests for the following scenarios.

### Dataset tests

- Valid image and annotation
- Missing annotation
- Invalid class ID
- Corrupt image
- Incorrect bounding-box coordinates

### Detection tests

- Image with a visible helmet
- Image with a visible worker and no helmet
- Multiple workers in one frame
- Occluded safety equipment
- Low-confidence detections
- Image with no workers

### Video tests

- Valid short video
- Unsupported format
- Corrupt video
- Video with no detections
- Video with multiple flagged frames
- Timestamp consistency
- Large file handling

### Database tests

- Insert and retrieve an incident
- Update a review status
- Filter incidents
- Export incident records
- Persist data across application restarts

### User interface tests

- Empty state
- Successful upload
- Failed upload
- Inference failure
- No incidents found
- Database error
- Download and export actions

---

# 11. PERFORMANCE AND SUCCESS CRITERIA

Do not promise a particular model accuracy before conducting experiments.

Use the following evaluation framework:

| Area | Measurement |
|---|---|
| Object detection | Precision, recall, mAP50 and mAP50–95 |
| PPE rules | Precision and recall on manually labeled test cases |
| Incident review | Number of confirmed, dismissed and uncertain events |
| Processing speed | Average inference latency and processed frames per second |
| Application reliability | Successful completion of defined workflows |
| Reproducibility | Ability to rerun evaluation using documented settings |

Establish baseline measurements first. Set improvement targets after observing actual results.

For video performance, distinguish model inference time from the complete application processing time, which also includes video decoding, annotation, storage and UI overhead.

---

# 12. SECURITY, PRIVACY AND RESPONSIBLE USE

- Process only media that the user is authorized to access.
- Use synthetic or public example videos for demonstrations.
- Do not upload confidential workplace footage to public hosting services without authorization.
- Validate file types and file sizes.
- Use safe temporary-file handling.
- Avoid exposing uploaded file paths or internal errors.
- Provide a method to delete uploaded media and associated records.
- Avoid face recognition and unnecessary identity tracking.
- Do not infer a worker's identity from appearance.
- Clearly communicate model uncertainty and possible false positives.
- Require human review before an event is treated as a confirmed violation.
- Document limitations arising from camera angle, lighting, occlusion and domain differences.

---

# 13. DEPLOYMENT PLAN

For the initial version:

1. Run training locally or in Google Colab.
2. Save the best checkpoint.
3. Test inference locally.
4. Deploy the Streamlit application on a compatible hosting service.
5. Include model download or model-path configuration.
6. Use environment variables for secrets if any are required.
7. Configure storage and database persistence.
8. Test the deployed application with sample images and videos.
9. Document resource limits and expected processing times.

Before selecting a hosting platform, verify its memory, storage, execution-time and upload-size limits. Video processing and model inference may exceed the resources available on free hosting tiers.

For a more reliable demonstration, a local application or a dedicated machine may be preferable.

---

# 14. REQUIRED DELIVERABLES

The completed repository must contain:

1. Working Streamlit application
2. Dataset acquisition and validation instructions
3. Model training script
4. Model evaluation script
5. Prediction and video-processing pipeline
6. Safety rule engine
7. SQLite incident database
8. Automated tests
9. Requirements file
10. Configuration example
11. README with setup instructions
12. System architecture diagram
13. Dataset documentation
14. Model card
15. Evaluation report with actual experimental results
16. Annotated example images
17. Demonstration video
18. Deployment instructions
19. Known limitations and future improvements

---

# 15. README STRUCTURE

Prepare the README with the following sections:

- Project title and overview
- Problem statement
- Key features
- System architecture
- Technology stack
- Dataset sources and licensing
- Environment setup
- Dataset preparation
- Model training
- Model evaluation
- Running the application
- Screenshots and demonstration
- Project structure
- Results
- Limitations
- Privacy and responsible use
- Future improvements
- Author and acknowledgements

Use actual screenshots and measured results from the completed application.

---

# 16. FINAL DEVELOPMENT PRIORITIES

Follow this priority order to avoid an unfinished project:

**Priority 1:** Working PPE detection on images.

**Priority 2:** Reliable, testable safety-rule logic.

**Priority 3:** Video processing with timestamps and evidence frames.

**Priority 4:** Persistent incident management and analytics dashboard.

**Priority 5:** Evaluation, testing and documentation.

**Priority 6:** Advanced temporal analysis, webcam support and deployment enhancements.

A polished and tested core system is more valuable than a large collection of incomplete features.

---

# 17. INSTRUCTIONS FOR ANTIGRAVITY

Act as a senior Machine Learning engineer and full-stack Python developer. Implement SafeSight AI according to this specification.

Follow these rules throughout development:

1. Inspect the current workspace before modifying files.
2. If the workspace is empty, create the proposed repository structure.
3. Implement the application incrementally, in the order of the development phases.
4. Use real, supported APIs and compatible dependencies.
5. Inspect the dataset configuration before relying on class names or annotations.
6. Never fabricate dataset contents, model metrics, training results or completed features.
7. Do not use random predictions or hardcoded incident records as if they were real model outputs.
8. Keep the detector, safety rules, database, analytics and UI modular.
9. Add error handling and tests as each component is implemented.
10. Prefer simple, maintainable code over unnecessary abstractions.
11. Do not build a complex real-time streaming system before image and video uploads work.
12. If a dataset or pretrained checkpoint is unavailable, provide clear setup instructions and report the blocker instead of silently substituting fake data.
13. Avoid committing large datasets, trained weights or confidential media.
14. Verify dependency compatibility and run the available tests.
15. Do not claim that the application is complete until its core workflows have been executed and verified.

## Required working process

Start by inspecting the environment, checking the installed Python version and available compute resources, and reviewing the dataset options.

Then:

- Create a short implementation checklist.
- Implement the smallest functional end-to-end version.
- Run it and test it.
- Fix errors before proceeding.
- Continue with the next phase.
- Maintain a progress document listing completed work, test results and remaining blockers.
- At completion, summarize how to install, train, run, test and deploy the project.

**First milestone:** Deliver a functioning application that accepts an image, runs genuine model inference, displays annotated detections and allows a supported potential PPE violation to be reviewed. Then extend it to video processing, incident storage and analytics.

The final result should be a reproducible, documented Deep Learning project that a student can understand, demonstrate and confidently explain in a technical interview.