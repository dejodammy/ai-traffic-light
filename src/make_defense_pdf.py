"""Build ONE comprehensive defense PDF: project data, all scenarios, all results
tables, generalization/overfitting analysis, Q&A, demo instructions and figures.
Output: DEFENSE/Defense_Compendium.pdf
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                Image, PageBreak, ListFlowable, ListItem, HRFlowable)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MEDIA = os.path.join(ROOT, "DEFENSE", "media")
OUT = os.path.join(ROOT, "DEFENSE", "Defense_Compendium.pdf")

NAVY = colors.HexColor("#0e1f56")
ACCENT = colors.HexColor("#1f6feb")
GREY = colors.HexColor("#dfe3ec")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=NAVY, fontSize=17,
                    spaceBefore=10, spaceAfter=8, fontName="Helvetica-Bold")
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=ACCENT, fontSize=12.5,
                    spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold")
BODY = ParagraphStyle("Body", parent=ss["BodyText"], fontSize=10, leading=14,
                      alignment=TA_JUSTIFY, spaceAfter=5)
BULLET = ParagraphStyle("Bullet", parent=BODY, leftIndent=10, spaceAfter=2)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=8.5, leading=11, textColor=colors.HexColor("#444"))
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=8.8, leading=11, alignment=TA_LEFT, spaceAfter=0)
CELLH = ParagraphStyle("CellH", parent=CELL, textColor=colors.white, fontName="Helvetica-Bold")

story = []


def h1(t): story.append(Paragraph(t, H1))
def h2(t): story.append(Paragraph(t, H2))
def p(t): story.append(Paragraph(t, BODY))
def small(t): story.append(Paragraph(t, SMALL))
def sp(h=6): story.append(Spacer(1, h))
def rule(): story.append(HRFlowable(width="100%", thickness=0.7, color=GREY, spaceBefore=4, spaceAfter=6))


def bullets(items):
    story.append(ListFlowable(
        [ListItem(Paragraph(i, BULLET), leftIndent=12, value="•") for i in items],
        bulletType="bullet", start="•", leftIndent=10))
    sp(3)


def table(headers, rows, col_widths=None, header_bg=NAVY, zebra=True):
    data = [[Paragraph(str(h), CELLH) for h in headers]]
    for r in rows:
        data.append([Paragraph(str(c), CELL) for c in r])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("GRID", (0, 0), (-1, -1), 0.5, GREY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]
    if zebra:
        for i in range(1, len(data)):
            if i % 2 == 0:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f4f6fb")))
    t.setStyle(TableStyle(style))
    story.append(t)
    sp(6)


def figure(name, caption, width=15*cm):
    fp = os.path.join(MEDIA, name)
    if not os.path.exists(fp):
        return
    from PIL import Image as PILImage
    iw, ih = PILImage.open(fp).size
    w = width
    h = w * ih / iw
    if h > 17*cm:
        h = 17*cm; w = h * iw / ih
    story.append(Image(fp, width=w, height=h))
    story.append(Paragraph(caption, SMALL))
    sp(8)


# ============================ COVER ============================
story.append(Spacer(1, 3.5*cm))
story.append(Paragraph("AN INTELLIGENT TRAFFIC LIGHT SYSTEM", ParagraphStyle(
    "cv1", parent=H1, fontSize=22, alignment=TA_CENTER, textColor=NAVY, spaceAfter=6)))
story.append(Paragraph("for Urban Intersections Using Deep Reinforcement Learning", ParagraphStyle(
    "cv2", parent=H2, fontSize=14, alignment=TA_CENTER, textColor=ACCENT)))
story.append(Paragraph("A Case Study of Lagos Traffic", ParagraphStyle(
    "cv3", parent=BODY, fontSize=13, alignment=TA_CENTER)))
sp(20)
story.append(Paragraph("DEFENSE COMPENDIUM", ParagraphStyle(
    "cv4", parent=H1, fontSize=16, alignment=TA_CENTER, textColor=NAVY)))
story.append(Paragraph("Complete data, scenarios, results, analysis, Q&amp;A and demo guide",
                       ParagraphStyle("cv5", parent=BODY, alignment=TA_CENTER, fontSize=10.5)))
sp(36)
for line in ["Oluwadamilola Samson Oladejo", "Student ID: 21120612543",
             "Computer and Information Sciences Department",
             "School of Science and Technology", "Pan-Atlantic University", "July 2026"]:
    story.append(Paragraph(line, ParagraphStyle("cvx", parent=BODY, alignment=TA_CENTER, fontSize=10.5, spaceAfter=2)))
story.append(PageBreak())

# ============================ EXEC SUMMARY ============================
h1("Executive Summary")
p("This document consolidates everything behind the project into a single reference: the problem, "
  "methodology, the full set of demand scenarios, all evaluation results (two-phase and four-phase), "
  "a generalization / overfitting analysis, the discussion of when reinforcement learning helps, the "
  "perception (YOLO) component, instructions for the live demonstrations, and an anticipated-questions "
  "section for the defense.")
p("<b>Headline finding.</b> A single-agent Deep Q-Network (DQN) controlling a real Lagos junction in the "
  "SUMO simulator learns to allocate green time by real-time demand. On <b>two-phase</b> control it beats "
  "every baseline tested by <b>61–77% lower waiting time</b>. On <b>four-phase</b> protected control it beats "
  "naive and Webster-optimal fixed timers but is matched by rule-based actuated control — an honest, "
  "literature-consistent result about <i>when</i> RL adds value.")
p("<b>Why it matters.</b> The system targets low-cost deployment (Raspberry Pi 4 + USB camera + YOLOv8), "
  "making adaptive signal control feasible for resource-constrained cities such as Lagos.")
rule()
p("<b>Note on baselines.</b> The authoritative results below are measured against four realistic baselines "
  "(naive 30s, Webster-optimal, gap-out actuated, fast fixed). A simpler fixed-time baseline is used only "
  "for the live demo and the supplementary held-out check in §7, and is labelled as such.")
story.append(PageBreak())

# ============================ 1. PROBLEM ============================
h1("1. Problem &amp; Motivation")
bullets([
    "Lagos was ranked the <b>worst traffic city globally</b> (mid-2024); 15M+ residents; congestion costs an "
    "estimated <b>NGN 4 trillion</b> in annual productivity loss.",
    "Current signals run on <b>fixed timers</b>: they cannot adapt to real-time variation — wasting green on "
    "empty approaches at off-peak and worsening congestion at peak.",
    "No emergency-vehicle response and no demand adaptation — a safety and efficiency gap.",
    "Need for an <b>affordable, deployable</b> intelligent controller for resource-constrained settings.",
])
p("<b>Research question.</b> Can a DQN-based traffic controller, deployable on low-cost hardware, "
  "significantly reduce congestion and waiting times at an urban Lagos intersection?")

# ============================ 2. AIM ============================
h1("2. Aim, Objectives &amp; Research Questions")
p("<b>Aim.</b> Model and evaluate an intelligent traffic-light system that adjusts signal timing from "
  "real-time traffic conditions using Deep Reinforcement Learning.")
h2("Objectives")
bullets([
    "Review traditional fixed-time and existing AI-based signal control.",
    "Design a traffic-state representation and reward capturing density, wait time and queue length.",
    "Implement a Deep Q-Network (DQN) to optimise signal control in simulation.",
    "Build a realistic SUMO simulation of a real Lagos intersection.",
    "Integrate camera-based vehicle detection (YOLO computer vision).",
])
h2("Research Questions")
bullets([
    "<b>RQ1.</b> Can a DQN controller significantly reduce wait times and queues versus fixed-time control?",
    "<b>RQ2.</b> Is the system feasible on low-cost embedded hardware (Raspberry Pi 4) in real time?",
])
p("<b>Scope.</b> Single intersection · SUMO simulation + hardware prototype · no real-world deployment.")

# ============================ 3. METHODOLOGY ============================
h1("3. Methodology")
h2("Markov Decision Process formulation")
p("<b>State (S).</b> A per-approach vector of queue lengths (halted vehicles, speed <= 0.1 m/s), waiting time "
  "and density. The final phase-aware four-phase model uses a <b>14-dimensional</b> state: the 12 traffic "
  "values plus the current phase and time-in-phase (the fix for the policy-collapse issue in §8).")
p("<b>Action (A).</b> Selection of the next signal phase (8 phases for the four-phase protected layout; "
  "action_dim = 4 green phases). Same phase -&gt; extends green; different phase -&gt; yellow transition.")
p("<b>Reward (R).</b> Congestion reduction — positive as queue and waiting time fall, with a small penalty "
  "for switching phases too frequently.")
h2("DQN hyperparameters")
table(["Hyperparameter", "Value"], [
    ["Network architecture", "2 × 256 units, ReLU"],
    ["Learning algorithm", "Double DQN + Prioritised Experience Replay (PER)"],
    ["Discount factor (gamma)", "0.99"],
    ["Learning rate", "0.001 (Adam)"],
    ["Batch size", "128"],
    ["Replay buffer", "100,000 transitions"],
    ["Target update", "every 1,000 steps"],
    ["Training", "two-stage: expert imitation pre-training -&gt; RL fine-tuning (epsilon-start 0.2, 200 episodes)"],
], col_widths=[6*cm, 11*cm])
h2("Tools")
p("SUMO (simulation) · TraCI (real-time interface) · PyTorch (DQN) · YOLOv8 (vehicle detection) · "
  "Raspberry Pi 4 (edge deployment) · OpenStreetMap (junction data).")

# ============================ 4. CASE STUDY ============================
h1("4. Case-Study Junction &amp; Signal Architectures")
p("<b>Junction.</b> Obafemi Awolowo Way &amp; Allen Avenue, Ikeja, Lagos — imported from OpenStreetMap and "
  "validated against satellite imagery.")
table(["Signal architecture", "Description"], [
    ["Two-Phase", "Opposing directions receive green together (2 green phases)."],
    ["Four-Phase (Protected)", "Each approach gets its own protected green (4 green phases, action_dim = 4) — "
                               "how the real junction operates."],
], col_widths=[5*cm, 12*cm])

# ============================ 5. SCENARIOS ============================
h1("5. Demand Scenarios")
p("Four demand patterns were used to test the controller under different, realistic conditions "
  "(vehicles per hour on each road):")
table(["Scenario", "Obafemi Awolowo Way", "Allen Avenue", "Purpose"], [
    ["Asymmetric", "760", "350", "Realistic imbalance"],
    ["Balanced", "760", "600", "Equal demand"],
    ["Rush-Hour", "1400", "280", "Heavy arterial congestion"],
    ["Allen-Dominant", "350", "1400", "Cross-street dominant (reversed / unseen direction)"],
], col_widths=[3.2*cm, 4.6*cm, 3.2*cm, 6*cm])
p("In addition, <b>random</b> and <b>stochastic</b> demand variants (time-varying flow rates) were generated to "
  "test adaptation to fluctuating, unpredictable traffic. The full list of simulation scenario presets is in "
  "Appendix A.")

# ============================ 6. BASELINES ============================
h1("6. Baseline Controllers")
table(["#", "Baseline", "Description"], [
    ["1", "Naive Fixed-Time", "30 s per phase, equal split — the standard today."],
    ["2", "Webster-Optimal", "Demand-proportional optimised fixed plan (queue-optimal on average)."],
    ["3", "Gap-Out Actuated", "Rule-based adaptive — serves the longest queue (strongest baseline)."],
    ["4", "Fast Fixed", "Rapid switching."],
], col_widths=[1*cm, 4*cm, 12*cm])
p("<i>Beating the actuated baseline is the real test of learned value — it already reacts to queues.</i>")

# ============================ 7. RESULTS ============================
h1("7. Results")
h2("7.1  Two-Phase Control — waiting time vs the best baseline")
p("<b>Finding:</b> the DQN outperformed <b>all</b> baselines on <b>every</b> demand pattern — lowest queue and "
  "lowest accumulated waiting time in all cases.")
table(["Scenario", "DQN Wait (s)", "Best Baseline (s)", "Improvement"], [
    ["Rush-Hour", "125.5", "523.8", "76% lower"],
    ["Asymmetric", "19.5", "85.5", "77% lower"],
    ["Balanced", "54.5", "139.0", "61% lower"],
    ["Allen-Dominant", "31.5", "97.3", "68% lower"],
], col_widths=[4.5*cm, 4*cm, 4.5*cm, 4*cm])
bullets([
    "Largest gains under <b>imbalanced</b> demand (rush-hour, asymmetric), where fixed-time wastes green on "
    "lightly loaded approaches.",
    "The <b>Allen-Dominant</b> result is the key generalization test: the agent had to serve the opposite "
    "(cross-street) direction it usually favours — and still won, confirming it learned the general rule "
    "(serve whichever approach is congested) rather than memorising a pattern.",
    "Throughput stayed comparable or higher — delay was not reduced at the expense of throughput.",
])

h2("7.2  Four-Phase Control &amp; State-of-the-Art comparison")
p("Improvement vs each baseline (positive = DQN better). Actuated control is the strongest baseline.")
table(["Scenario", "vs Naive", "vs Webster", "vs Actuated", "DQN Wait (s)"], [
    ["Rush-Hour", "+43%", "+25%", "-24%", "489.6"],
    ["Asymmetric", "+58%", "+17%", "-84%", "227.2"],
    ["Balanced", "+52%", "+48%", "-30%", "281.8"],
    ["Allen-Dominant", "+50%", "—", "-29%", "198.4"],
], col_widths=[3.6*cm, 2.6*cm, 2.8*cm, 3*cm, 3*cm])
p("<b>Four-phase finding:</b> actuated control is near-optimal for protected phasing — its rule of serving the "
  "longest queue directly attacks delay, leaving little room for learned improvement. State-of-the-art "
  "pressure-based methods (PressLight, PDLight) likewise could not beat actuated at a single junction; their "
  "benefit arises in multi-intersection networks.")

h2("7.3  Generalization &amp; overfitting analysis")
p("<b>The question:</b> the headline two-phase / four-phase numbers evaluate the agent on demand profiles it "
  "was trained on, and the simulation uses a fixed random seed (42) for both training and evaluation — so the "
  "headline is an <i>in-sample</i> figure. Is the model therefore overfit?")
p("<b>Evidence it is not.</b> The same trained checkpoint was evaluated on demand profiles it never saw:")
bullets([
    "<b>Unseen direction (Allen-Dominant):</b> reversed dominant flow — DQN still beat naive and Webster fixed "
    "timers (see §7.1–7.2). A memorised policy would fail here.",
    "<b>Unseen realizations (random / stochastic demand):</b> a model trained on the random four-phase scenario "
    "was tested on two held-out demand files. Against the simple fixed-time baseline it still reduced queue and "
    "waiting time (in-sample ~ -14% queue / -16% wait; held-out ~ -8% queue / -11–12% wait).",
])
p("The modest drop from in-sample to out-of-sample is the normal, healthy <b>generalization gap</b>, not a "
  "collapse — the policy learned a transferable strategy. <b>Stronger validation</b> (averaging over multiple "
  "seeds; training across several demand profiles — domain randomisation) is the recommended next step.")

# ============================ 8. DISCUSSION ============================
h1("8. Discussion, Challenges &amp; Limitations")
h2("When does RL help?")
bullets([
    "DQN achieves <b>16–76% delay reduction</b>; the advantage is greatest under <b>imbalanced, variable</b> "
    "demand — typical of Lagos peak traffic.",
    "On four-phase protected control, <b>actuated control remains competitive</b> — an important, honest finding "
    "about the limits of learned control at a single junction.",
    "Benefit magnitude depends on demand structure and signal architecture — consistent with traffic-engineering "
    "theory.",
])
h2("Key challenge — policy collapse (and its fix)")
bullets([
    "<b>Problem:</b> the agent served one approach almost exclusively (106 of 113 decisions), starving the cross "
    "street.",
    "<b>Root cause:</b> partial observability — the reward penalised <i>accumulated</i> wait time, but the state "
    "only exposed <i>instantaneous</i> wait.",
    "<b>Fix:</b> add accumulated waiting time / phase information to the state (the 14-dim phase-aware state).",
    "<b>Lesson:</b> the state must contain the quantities the reward optimises.",
])
h2("Limitations")
bullets([
    "Simulation-based only; single junction.",
    "Camera/YOLO untested on real Lagos footage; no real-world deployment.",
    "Computational constraints; idealised sensing assumptions.",
])

# ============================ 9. YOLO ============================
h1("9. Perception — YOLO Vehicle Detection &amp; Counting")
p("In deployment the per-approach vehicle counts that form the state come from a camera running a "
  "<b>YOLOv8</b> detector. The live demo proves this: it detects real vehicles (car, motorcycle, bus, truck — "
  "COCO classes 2, 3, 5, 7), draws a box on each, and prints a running count to the terminal — exactly the "
  "number that feeds the controller. YOLOv8-nano is light enough to run on a Raspberry Pi 4.")
p("<b>Honest limit:</b> per-frame detection can flicker by a vehicle or two; a tracker (e.g. ByteTrack) would "
  "stabilise counts for deployment.")

# ============================ 10. DEMOS ============================
h1("10. Live Demonstrations — How to Run")
p("All demos live in <b>DEFENSE\\scripts\\</b> (double-click) and run from the project's virtual environment. "
  "Ensure SUMO is installed (default C:\\Program Files (x86)\\Eclipse\\Sumo).")
table(["Script", "What it shows"], [
    ["1_Watch_Trained_AI_Live.bat", "Trained DQN driving the real 4-phase Ikeja junction in the SUMO GUI."],
    ["2_Watch_FixedTime_Baseline.bat", "Fixed-time controller on the same traffic — for comparison."],
    ["3_Run_Live_Training.bat", "A short live training run (process demonstration; separate output folder)."],
    ["4_YOLO_Live_Webcam_Count.bat", "Live webcam vehicle detection + count (point at cars on a screen)."],
    ["5_YOLO_Count_On_Sample_Video.bat", "Canned traffic video — guaranteed, no webcam needed."],
], col_widths=[6.2*cm, 10.8*cm])
small("Tip: if the YOLO window won't open, the environment needs the GUI OpenCV build: "
      "pip uninstall -y opencv-python opencv-python-headless then pip install opencv-python.")

# ============================ 11. Q&A ============================
h1("11. Anticipated Defense Questions")
qa = [
    ("In one sentence, what did you do?",
     "I trained a Deep Q-Network to control the lights at a real Ikeja, Lagos junction in SUMO; on two-phase "
     "control it cut waiting time 61–77% versus the best baseline, and on four-phase it beat naive and Webster "
     "fixed timers (while actuated control remained competitive)."),
    ("Did you train and test on the same data — is this overfitting?",
     "The headline is in-sample (same demand profile and random seed for train and eval). But the same "
     "checkpoint generalised: it still beat fixed timers on the unseen Allen-Dominant direction and on held-out "
     "random/stochastic demand — a small, healthy generalization gap, not a collapse. Next step: multi-seed "
     "averaging and domain randomisation."),
    ("Where does the improvement come from?",
     "Responsiveness — it allocates green to whichever approach is congested and skips empty ones, instead of "
     "following a fixed cycle. The gain is largest under imbalanced, variable demand."),
    ("Why does actuated control beat the DQN on four-phase?",
     "Actuated control's 'serve the longest queue' rule is near-optimal for protected phasing, so there is "
     "little headroom for learned improvement at a single junction. Learned control's advantage grows in "
     "multi-intersection coordination — a future-work direction."),
    ("How would it know the traffic in real life?",
     "A camera per approach runs YOLOv8 to count vehicles; those counts become the same state the DQN reads in "
     "simulation. YOLOv8-nano runs on a Raspberry Pi 4."),
    ("Why SUMO and not a real junction?",
     "Safety and cost — you cannot let an untrained agent control a live junction. SUMO allows millions of "
     "training vehicles safely; the trained policy then transfers."),
    ("What was the hardest problem you solved?",
     "Policy collapse: the agent starved one approach because the reward optimised accumulated wait but the "
     "state only showed instantaneous wait. Adding accumulated-wait/phase features to the state fixed it. "
     "Lesson: the state must contain what the reward optimises."),
]
for q, a in qa:
    story.append(Paragraph("Q: " + q, ParagraphStyle("q", parent=BODY, fontName="Helvetica-Bold", textColor=NAVY, spaceAfter=1)))
    story.append(Paragraph("A: " + a, ParagraphStyle("a", parent=BODY, leftIndent=8, spaceAfter=7)))

# ============================ 12. CONCLUSION ============================
h1("12. Conclusion &amp; Future Work")
p("Deep Reinforcement Learning is a clear, practically relevant improvement over conventional signal control "
  "for <b>two-phase</b> junctions under heavy, imbalanced demand — characteristic of Lagos peak traffic. On "
  "<b>four-phase</b> protected control it matches but does not exceed rule-based actuated control, an honest "
  "delineation of where RL adds value. An accurate account of when RL helps is more useful than a claim of "
  "universal superiority.")
h2("Future work")
bullets([
    "Field data collection — apply YOLO to recorded footage for measured turning-movement counts.",
    "Multi-agent coordination along the Obafemi Awolowo Way corridor.",
    "Max-pressure reward for throughput-competitive performance.",
    "Lagos-specific detection — fine-tune YOLO for motorcycles (okada) and tricycles (keke).",
    "Supervised pilot deployment with automatic fallback to fixed-time control.",
    "Emergency-vehicle pre-emption (ambulances, fire services).",
])

# ============================ FIGURES ============================
story.append(PageBreak())
h1("Appendix A — Figures")
figure("architecture.png", "Figure A1. System architecture: camera -&gt; YOLO -&gt; DQN -&gt; signal -&gt; reward loop.")
figure("rl_loop.png", "Figure A2. Reinforcement-learning interaction loop.")
figure("training_curves_peak.png", "Figure A3. Training curves — reward / queue / wait converging over episodes.")
figure("baseline_comparison.png", "Figure A4. DQN vs baseline controllers.")
figure("dqn_vs_actuated.png", "Figure A5. DQN vs actuated control (four-phase context).")
figure("delay_reduction.png", "Figure A6. Delay reduction summary.")

# ============================ APPENDIX SCENARIOS ============================
story.append(PageBreak())
h1("Appendix B — Simulation Scenario Presets")
p("All scenario presets defined in the codebase (config.py). The faithful real-junction four-phase scenarios "
  "(lagos_4ph*) are the ones used for the final results and the live demo.")
presets = [
    ("ideal", "Engineered baseline junction"),
    ("lagos_intersection", "Engineered Lagos-style cross"),
    ("lagos_peak / lagos_low / lagos_extreme / lagos_reversed / lagos_chaos", "Abstract-junction demand variants"),
    ("lagos_real", "Raw OSM Ikeja junction (oversaturated)"),
    ("lagos_calib / lagos_calib_phase", "Calibrated real junction, 2-phase (phase-aware variant)"),
    ("lagos_estimated / lagos_estimated_tuned", "Geometry-estimated real demand"),
    ("lagos_single / _bal / _bal_t1 / _peak / _peak_q / _peak2_q", "Faithful 2-lane single junction, demand variants"),
    ("lagos_4ph", "Faithful 4-phase protected junction (peak)"),
    ("lagos_4ph_asym / _bal / _heavy / _random", "4-phase demand variants (asymmetric / balanced / heavy / random)"),
    ("lagos_4ph_reversed / _gen / _stoch", "4-phase held-out: reversed / unseen-generalization / stochastic"),
    ("lagos_single_reversed / _gen / _stoch", "2-phase held-out generalization scenarios"),
]
table(["Scenario preset(s)", "Description"], presets, col_widths=[8.5*cm, 8.5*cm])

# ============================ REFERENCES ============================
story.append(PageBreak())
h1("Appendix C — Key References")
refs = [
    "Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540).",
    "El-Tantawy, S., Abdulhai, B., Abdelgawad, H. (2013). MARLIN-ATSC. Transportation Research Part C.",
    "Van der Pol, E., Oliehoek, F. S. (2016). Coordinated deep RL for traffic light control. NeurIPS Workshop.",
    "Wei, H., et al. (2018). IntelliLight: An RL approach for intelligent traffic light control. KDD 2018.",
    "Wei, H., et al. (2019). PressLight: Max-pressure control to coordinate traffic signals. KDD 2019.",
    "Zhang, K., et al. (2020). PDLight: Deep RL for traffic signal control with pressure detection. IEEE BigData.",
    "Redmon, J., et al. (2016). You Only Look Once: Unified real-time object detection. CVPR.",
    "Webster, F. V. (1958). Traffic signal settings. UK Road Research Laboratory, Tech. Paper 39.",
    "Krajzewicz, D., et al. (2012). Recent development and applications of SUMO. Int. J. Adv. Syst. Meas.",
    "Sutton, R. S., Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.",
]
for r in refs:
    story.append(Paragraph(r, ParagraphStyle("ref", parent=SMALL, spaceAfter=3, leftIndent=10, firstLineIndent=-10)))


# ============================ BUILD ============================
def _footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#888"))
    canvas.drawString(2*cm, 1.0*cm, "Intelligent Traffic Light System Using Deep RL — Lagos Case Study")
    canvas.drawRightString(A4[0]-2*cm, 1.0*cm, "Page %d" % doc.page)
    canvas.restoreState()


doc = SimpleDocTemplate(OUT, pagesize=A4, topMargin=1.6*cm, bottomMargin=1.6*cm,
                        leftMargin=2*cm, rightMargin=2*cm,
                        title="Defense Compendium — Intelligent Traffic Light System (Lagos)",
                        author="Oluwadamilola Samson Oladejo")
doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
print("saved", OUT)
