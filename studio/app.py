"""SurgVision Studio — 수술영상 AI 연구를 아이디어에서 원고까지 따라가는 로컬 웹앱.

실행:  streamlit run app.py   (또는 start_studio.command 더블클릭)
데이터는 이 컴퓨터의 ~/SurgVisionProjects/<프로젝트>/ 에만 저장됩니다.
"""
from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image

import core as C

st.set_page_config(page_title="SurgVision Studio", page_icon="🩺", layout="wide")

# ------------------------------------------------------------------ sidebar
st.sidebar.title("SurgVision Studio")
st.sidebar.caption("아이디어 + 수술영상 → 설계 · 데이터 · 아노테이션 · 모델 · 평가 · 외부검증 · 원고")
projects = C.list_projects()
NAMES = [name for _, name in C.STEPS]
if "pending_project" in st.session_state:          # set by a save in the previous run
    st.session_state["proj_sel"] = st.session_state.pop("pending_project")
if st.session_state.get("proj_sel") not in ["(새 프로젝트)"] + projects:
    st.session_state["proj_sel"] = "(새 프로젝트)"
sel = st.sidebar.selectbox("프로젝트", ["(새 프로젝트)"] + projects, key="proj_sel")
slug = None if sel == "(새 프로젝트)" else sel
proj = C.pdir(slug) if slug else None
done = C.status(proj) if proj else {}
st.session_state.setdefault("nav", NAMES[0])
choice = st.sidebar.radio("단계", NAMES, key="nav")
page_i = NAMES.index(choice)
page = C.STEPS[page_i][0]
st.sidebar.caption("진행: " + " ".join(("✅" if done.get(k) else "⬜") + n.split(".")[0] for k, n in C.STEPS[1:]))
st.sidebar.divider()
st.sidebar.caption(f"저장 위치: {C.PROJECTS}")
st.sidebar.caption("이 결과 모델은 의료기기나 임상 판단 도구가 아닙니다.")

pack = C.rj(proj / "pack.json") if proj else None


def need_project():
    if not proj:
        st.info("먼저 **1. 아이디어 → 설계**에서 프로젝트를 만드세요.")
        st.stop()


def log_area():
    return st.empty()


def show_md(path: Path):
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))


def _goto(i):
    st.session_state["nav"] = NAMES[min(i, len(NAMES) - 1)]


def go_next():
    st.button("다음 단계로 →", type="primary", on_click=_goto, args=(page_i + 1,))


# ================================================================== 0. home
if page == "home":
    st.title("수술영상 AI 연구를 처음부터 끝까지")
    st.markdown("""
아이디어와 수술 영상만 있으면 왼쪽 단계를 위에서부터 차례로 진행합니다. 각 단계는 **실제로 파일을 만들고**, 만든 파일이 있으면 ✅로 바뀝니다.

| 단계 | 하는 일 | 결과 파일 |
|---|---|---|
| 1 설계 | 수술·질문·과제(분할/분류)·클래스 정의·경계 규칙·지표 | `pack.json`, `config.json` |
| 2 필요한 건수 | 시험 환자 수(신뢰구간 폭) 계산, 나중에 학습곡선 | `sample_size/` |
| 3 거버넌스 | IRB·외부 반출·비식별 결정 | `governance_decision.json` |
| 4 데이터 | 영상 → 비식별 프레임, 또는 라벨 있는 공개 데이터 가져오기 | `frames/`, `labeled/` |
| 5 화면 검토 | 체내 밖·식별 가능 화면 사람이 제외 | 검토 기록 |
| 6 아노테이션 | 분할: CVAT 왕복 / 분류: 앱 안 태거·구간표 / 판독자 간 일치도 | `labeled/labeled_manifest.csv` |
| 7 데이터 구성 | 환자 단위 분할·누수 검사·클래스 균형 점검 | `split/`, `audit/` |
| 8 모델 후보 | 여러 구조를 **검증 자료로만** 비교해 선택 | `search/` |
| 9 학습·평가 | 선택 모델 학습 → 시험 1회 → 그림·영상 / 학습곡선 | `runs/<시각>/` |
| 10 외부검증 | 다른 기관·공개 데이터에 고정 모델 적용 | `external/` |
| 11 원고 | 숫자마다 출처가 달린 원고 초안 + 결과 묶음 다운로드 | `manuscript_draft.md` |

**지켜지는 원칙:** 원본 영상과 대응표는 이 컴퓨터 밖으로 나가지 않음 · 아노테이션 없는 프레임은 음성이 아님 · 환자 단위 분할 · 시험 자료는 마지막에 한 번 · 떨어진 프레임에 보간 금지 · 시험 환자 5명 미만이면 feasibility.
""")
    if proj:
        st.subheader(f"현재 프로젝트: {pack['title_ko']}")
        st.json({k: v for k, v in done.items()}, expanded=False)
    st.markdown("**예제로 먼저 해보기:** 실습 키트의 `example_videos/`(공개 CholecSeg8k로 만든 모의 영상 5개)와 "
                "`example_reference_labels.zip`을 쓰면 IRB 없이 전 과정을 연습할 수 있습니다.")
    go_next()

# ================================================================== 1. design
elif page == "design":
    st.title("1. 아이디어 → 연구 설계")
    st.caption("어떤 수술이든 됩니다. 질문을 '무엇을 찾거나 나누는가'로 바꾸는 단계입니다.")
    p = pack or {}
    with st.form("design"):
        c1, c2 = st.columns(2)
        title = c1.text_input("연구 제목(한글)", p.get("title_ko", ""), placeholder="예: 로봇 전립선절제술 영상의 신경혈관다발 분할")
        title_en = c2.text_input("영문 제목(원고용, 선택)", p.get("title_en", ""))
        procedure = c1.text_input("수술 종류", p.get("data", {}).get("procedure", ""), placeholder="예: robot-assisted radical prostatectomy")
        mod_lab = c2.selectbox("영상 종류", list(C.MODALITIES.values()),
                               index=list(C.MODALITIES).index(p.get("data", {}).get("modality", "laparoscopic")))
        modality = {v: k for k, v in C.MODALITIES.items()}[mod_lab]
        question = st.text_area("연구 질문 (대상·과제·평가 단위가 들어가게)", p.get("research_question", ""),
                                placeholder="예: RARP 영상에서 모델이 화면에 보이는 신경혈관다발을 환자 단위로 분리된 시험 환자에서 얼마나 정확히 분할하는가?")
        rationale = st.text_input("임상적 이유(1문장)", p.get("clinical_rationale", ""))
        st.markdown("**과제 유형 고르기** — '어디에 있나(경계)' → 분할 · '이 장면은 무엇인가/기준을 만족하나' → 프레임 태그 분류 · '언제부터 언제까지(phase)' → 구간 분류")
        MODE = {"분할(구조물·기구)": "seg", "프레임 분류(태그)": "tag", "phase 구간 분류": "interval"}
        mode = MODE[st.radio("과제", list(MODE), horizontal=True,
                             index=list(MODE.values()).index(p.get("annotation", {}).get("mode", "seg")))]
        prev_cls = [c for c in p.get("class_names", []) if c != "background"]
        classes = st.text_input("클래스 이름(영문 소문자, 쉼표로) — CVAT 라벨과 글자까지 같아야 함",
                                ", ".join(prev_cls), placeholder="예: nvb  또는  cvs_achieved, cvs_not_achieved, not_assessable")
        defs = st.text_area("클래스별 정의 — 한 줄에 '클래스: 정의'", "\n".join(f"{k}: {v}" for k, v in p.get("label_definitions", {}).items()),
                            placeholder="nvb: 화면에 실제로 보이는 신경혈관다발. 가려진 부분 추정은 하지 않는다.")
        hard = st.text_area("경계 판단 규칙 — 한 줄에 하나, 3개 이상", "\n".join(p.get("hard_cases", [])),
                            placeholder="기구가 가리면 기구 부분은 칠하지 않는다\n연기로 경계가 안 보이면 판단 보류\n...")
        c3, c4, c5 = st.columns(3)
        absent = c3.checkbox("대상이 안 보이는 프레임이 흔하다", p.get("_absent", True))
        small = c4.checkbox("대상이 작다(화면의 5% 미만)", p.get("_small", False))
        rare = c5.checkbox("드물지만 놓치면 안 되는 클래스가 있다", p.get("_rare", False))
        c6, c7, c8, c9 = st.columns(4)
        SRC = {"우리 병원 영상": "own_video", "공개 데이터": "public"}
        source = SRC[c6.selectbox("데이터", list(SRC), index=0 if p.get("data", {}).get("source", "own_video") == "own_video" else 1)]
        n_pat = c7.number_input("계획 환자 수", 3, 5000, int(p.get("data", {}).get("n_patients_planned", 20)))
        fps = c8.number_input("프레임 추출 (초당)", 0.1, 25.0, float(p.get("data", {}).get("extract_fps", 1.0)))
        annotators = c9.number_input("판독자 수", 1, 5, int(p.get("annotation", {}).get("annotators", 1)))
        c10, c11, c12 = st.columns(3)
        fpv = c10.number_input("영상당 아노테이션 장수(분할/태그)", 3, 500, int(p.get("annotation", {}).get("frames_per_video", 20)))
        gap = c11.number_input("프레임 최소 간격(초)", 0.0, 600.0, float(p.get("annotation", {}).get("min_gap_sec", 5)))
        dbl = c12.number_input("이중판독 비율(일치도용)", 0.0, 1.0, float(p.get("annotation", {}).get("double_annotation_fraction", 0.0)))
        ok = st.form_submit_button("설계 저장", type="primary")
    if ok:
        cls = [c.strip() for c in classes.split(",") if c.strip()]
        dd = {}
        for line in defs.splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                dd[k.strip()] = v.strip()
        hc = [h.strip() for h in hard.splitlines() if h.strip()]
        errs = []
        if not (title and procedure and question and cls):
            errs.append("제목·수술·질문·클래스는 필수입니다.")
        if any(not C.re.fullmatch(r"[a-z0-9_]+", c) for c in cls):
            errs.append("클래스 이름은 영문 소문자·숫자·_ 만.")
        if mode != "seg" and len(cls) < 2:
            errs.append("분류는 클래스가 2개 이상이어야 합니다.")
        miss = [c for c in cls if c not in dd]
        if miss:
            errs.append(f"정의가 없는 클래스: {miss}")
        if len(hc) < 3:
            errs.append("경계 규칙을 3개 이상 적어 주세요. 칠할 때 망설이는 지점입니다.")
        if errs:
            for e in errs:
                st.error(e)
        else:
            s = slug or C.slugify(title_en or procedure + "_" + "_".join(cls[:2]))
            newp = C.new_pack(slug=s, title_ko=title, title_en=title_en, question=question, rationale=rationale,
                              task_type="segmentation" if mode == "seg" else "classification", mode=mode, classes=cls,
                              definitions=dd, hard_cases=hc, source=source, procedure=procedure, modality=modality,
                              n_patients=int(n_pat), fps=float(fps), frames_per_video=int(fpv), min_gap=float(gap),
                              annotators=int(annotators), double_fraction=float(dbl),
                              absent_common=absent, small_structure=small, rare_important=rare)
            newp.update({"_absent": absent, "_small": small, "_rare": rare})
            if pack:   # keep later decisions
                newp["split"], newp["governance"] = pack.get("split", newp["split"]), pack.get("governance", newp["governance"])
                newp["training"]["arch"] = pack["training"].get("arch", newp["training"]["arch"])
            pd = C.pdir(s)
            pd.mkdir(parents=True, exist_ok=True)
            C.wj(pd / "pack.json", newp)
            C.write_config(pd, newp)
            st.session_state["pending_project"] = s
            slug, proj, pack = s, pd, newp
            st.success(f"저장했습니다: {pd} — 아래 '다음 단계로'를 누르세요.")
    if pack:
        st.subheader("설계 요약과 추천")
        m = pack["metrics"]
        st.markdown(f"- **과제:** {pack['task_type']} / {pack['annotation']['mode']} · **클래스:** {', '.join(pack['class_names'])}\n"
                    f"- **1차 지표:** {m['primary']}\n- **보조 지표:** {', '.join(m['secondary'])}\n- **이유:** {m['rationale']}\n"
                    f"- **아노테이션 방법:** {C.recommend(pack['task_type'], pack['annotation']['mode'], False, False, False)['annotation']}\n"
                    f"- **모델 후보(8단계에서 비교):** {', '.join(m['candidates'])}")
        budget = {"seg": 2.0, "tag": 0.15, "interval": None}[pack["annotation"]["mode"]]
        if budget:
            n = pack["data"]["n_patients_planned"] * pack["annotation"]["frames_per_video"]
            st.info(f"아노테이션 예상: {n:,}장 × 약 {budget}분 ≈ **{n * budget / 60:.1f}시간** (판독자 1명 기준). 너무 길면 영상당 장수를 줄이세요.")
        st.caption("더 깊은 설계 검토가 필요하면 Claude Code의 surgvision-loop 하네스나 Aside S1 프롬프트를 쓰세요.")
        go_next()

# ================================================================== 2. size
elif page == "size":
    need_project()
    st.title("2. 의미 있는 건수 정하기")
    st.markdown("""
두 가지를 따로 봅니다.
1. **시험(평가)에 필요한 환자 수** — 결과의 95% 신뢰구간을 얼마나 좁게 할지로 정합니다. 같은 환자의 프레임끼리는 비슷하므로 *설계효과*로 보정합니다.
2. **학습에 필요한 환자 수** — 미리 알 수 없습니다. 파일럿을 돌린 뒤 **9단계의 학습곡선**으로 '몇 명 더 모으면 얼마나 좋아지나'를 추정합니다.
""")
    seg = pack["task_type"] == "segmentation"
    c1, c2, c3 = st.columns(3)
    metric = "dice" if seg else "proportion"
    expected = c1.number_input("예상 성능(Dice 또는 정확도/민감도)", 0.05, 0.99, 0.80 if seg else 0.85)
    sd = c2.number_input("프레임 간 표준편차(Dice만)", 0.01, 0.5, 0.15) if seg else None
    icc = c3.number_input("같은 환자 프레임 간 상관(ICC)", 0.0, 1.0, 0.3, help="모르면 0.2–0.4. 파일럿 결과가 있으면 아래에서 실제 값으로 계산")
    c4, c5, c6 = st.columns(3)
    m = c4.number_input("환자당 시험 프레임 수", 1, 1000, int(pack["annotation"]["frames_per_video"]))
    hw = c5.number_input("허용 신뢰구간 반폭(±)", 0.01, 0.3, 0.05)
    prev = c6.number_input("관심 클래스 비율(민감도일 때)", 0.0, 1.0, 0.0) if not seg else 0.0
    run = C.latest_run(proj)
    use_pilot = False
    if run and seg and (run / "per_case_test.csv").exists():
        use_pilot = st.checkbox("최근 학습 결과(파일럿)에서 SD·ICC 추정하기", True)
    if st.button("계산", type="primary"):
        args = ["precision", "--metric", metric, "--expected", expected, "--icc", icc, "--frames-per-patient", m,
                "--half-width", hw, "--out", proj / "sample_size"]
        if seg:
            args += ["--sd", sd]
        if prev:
            args += ["--prevalence", prev]
        if use_pilot:
            args += ["--pilot", run / "per_case_test.csv", "--column", f"dice_{pack['class_names'][1]}"]
        rc, out = C.run_tool("sample_size.py", args, proj)
        if rc:
            st.error(out[-800:])
    r = C.rj(proj / "sample_size" / "sample_size_precision.json")
    if r:
        st.metric("필요한 시험 환자 수", r["test_patients_needed"])
        st.write(f"설계효과 {r['design_effect']} (SD {r['sd']}, ICC {r['icc']}, 출처: {r['source']}) · 필요한 프레임 {r['frames_needed']}")
        st.write("환자당 프레임 수를 바꾸면:", r["what_if_frames_per_patient"])
        if r.get("note_prevalence"):
            st.warning(r["note_prevalence"])
        tot = pack["data"]["n_patients_planned"]
        ratio = float(pack["split"]["ratios"].split(",")[2])
        st.info(f"현재 계획: 환자 {tot}명 × 시험 비율 {ratio} = 시험 약 {round(tot * ratio)}명. "
                + ("충분합니다." if round(tot * ratio) >= r["test_patients_needed"] else "부족합니다 — 환자를 늘리거나 결과를 feasibility로 보고하세요."))
    go_next()

# ================================================================== 3. governance
elif page == "gov":
    need_project()
    st.title("3. 거버넌스·비식별")
    st.markdown("클라우드 CVAT·Colab은 **외부 서버**입니다. 어디까지 보낼 수 있는지 데이터에 손대기 전에 정합니다. "
                "확실하지 않으면 '아니오'로 두세요 — 나중에 허용되면 바꾸면 되지만, 잘못 올린 프레임은 되돌릴 수 없습니다.")
    g = C.rj(proj / "governance_decision.json", {})
    public = pack["data"]["source"] == "public"
    with st.form("gov"):
        if public:
            irb = "public_data"
            lic = st.text_input("공개 데이터 이름·라이선스(예: CholecSeg8k, CC BY-NC-SA 4.0)", g.get("license", ""))
        else:
            IRB = {"아직/모름": "pending", "승인": "approved", "심의면제 확인": "exempt_confirmed"}
            irb = IRB[st.selectbox("IRB 상태", list(IRB), index=list(IRB.values()).index(g["irb_status"])
                                   if g.get("irb_status") in IRB.values() else 0)]
            lic = ""
        evidence = st.text_input("근거 문서명·보관 위치(번호는 적지 않음)", g.get("irb_evidence", ""))
        cloud = st.radio("비식별 프레임을 외부 클라우드(app.cvat.ai, Colab)에 올리는 것이 문서로 허용되나요?",
                         ["아니오/모름", "예"], index=1 if g.get("compute_platform") == "cloud_approved" or public else 0, horizontal=True)
        burned = st.text_input("화면에 박힌 글자(이름·번호·날짜) 위치 — 없으면 '없음'", g.get("burned_in_text", ""))
        custodian = st.text_input("대응표(linkage) 보관 책임자 — 역할만", g.get("linkage_custodian", ""))
        oq = st.text_area("IRB/정보보호팀에 확인할 질문", "\n".join(g.get("open_questions", [])))
        who = st.text_input("결정자 이니셜", g.get("decided_by", ""))
        sub = st.form_submit_button("결정 저장", type="primary")
    if sub:
        if not who:
            st.error("결정자 이니셜을 적어 주세요.")
        else:
            dec = {"irb_status": irb, "irb_evidence": evidence, "license": lic,
                   "data_class": "public_dataset" if public else "deidentified_frames",
                   "compute_platform": "cloud_approved" if cloud == "예" else "local_only",
                   "annotation_platform": ("cvat_cloud" if cloud == "예" else "cvat_selfhosted") if pack["annotation"]["mode"] == "seg" else "in_app",
                   "burned_in_text": burned, "linkage_custodian": custodian,
                   "open_questions": [q for q in oq.splitlines() if q.strip()], "decided_by": who,
                   "decided_on": C.dt.date.today().isoformat()}
            C.wj(proj / "governance_decision.json", dec)
            pack["governance"]["irb_status"] = irb
            C.wj(proj / "pack.json", pack)
            g = dec
            st.success("저장했습니다.")
    if g.get("decided_by"):
        ok, why = C.gov_allows_data(proj)
        (st.success if ok else st.warning)("데이터 처리 가능" if ok else why)
        if g["compute_platform"] == "local_only" and pack["annotation"]["mode"] == "seg":
            st.info("분할 아노테이션은 원내 self-hosted CVAT를 쓰세요(설치: https://docs.cvat.ai/docs/administration/basics/installation/). "
                    "분류는 이 앱 안에서 하므로 외부로 나가지 않습니다.")
        st.markdown("**비식별 네 겹:** 파일명→연구번호 · 메타데이터 제거 · 화면 글자 가림 · 체내 밖/식별 화면 사람 전수 검토(5단계)")
    go_next()

# ================================================================== 4. data
elif page == "data":
    need_project()
    st.title("4. 데이터 넣기")
    ok, why = C.gov_allows_data(proj)
    if not ok:
        st.warning(why)
        st.stop()
    tab1, tab2 = st.tabs(["A. 수술 영상 → 프레임", "B. 이미 라벨 있는 이미지 폴더(공개 데이터 등)"])
    with tab1:
        st.markdown("영상은 **이 컴퓨터의 폴더 경로**로 지정합니다(업로드하지 않음). 파일명은 연구번호로 바뀌고 원래 이름은 대응표에만 남습니다.")
        vdir = st.text_input("원본 영상 폴더 경로", st.session_state.get("vdir", ""), placeholder="/Users/.../raw_videos")
        st.session_state["vdir"] = vdir
        vids = sorted(p for p in Path(vdir).glob("*") if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".wmv"}) if vdir and Path(vdir).is_dir() else []
        if vids:
            link = proj / "linkage.csv"
            given = Path(vdir) / "linkage.csv"
            old = {r["original_file"]: r for r in C.read_csv(link)} if link.exists() else (
                {r["original_file"]: r for r in C.read_csv(given)} if given.exists() else {})
            if given.exists() and not link.exists():
                st.caption("영상 폴더의 linkage.csv를 불러왔습니다.")
            rows = [old.get(v.name, {"original_file": v.name, "patient_id": f"P{i + 1:03d}", "video_id": f"V{i + 1:03d}"})
                    for i, v in enumerate(vids)]
            st.markdown("**대응표** — 같은 환자의 영상이 여러 개면 patient_id를 같게 하세요. 시간적 검증을 원하면 수술일 순으로 번호를 매기세요.")
            ed = st.data_editor(rows, num_rows="fixed", use_container_width=True, key="linkage")
            c1, c2 = st.columns(2)
            blackout = c1.text_input("화면 글자 가리기 x,y,w,h (원본 픽셀; 여러 개는 ;)", pack["data"].get("blackout", ""))
            maxw = c2.number_input("프레임 최대 너비(px)", 320, 1920, 854)
            if st.button("프레임 추출 시작", type="primary"):
                C.write_csv(list(ed), link)
                pack["data"]["blackout"] = blackout
                C.wj(proj / "pack.json", pack)
                args = ["--videos", vdir, "--linkage", link, "--out", proj / "frames", "--fps", pack["data"]["extract_fps"],
                        "--max-width", maxw, "--modality", pack["data"].get("modality", "laparoscopic")]
                if blackout:
                    args += ["--blackout", blackout]
                with st.status("추출 중…", expanded=True):
                    rc, out = C.run_tool("extract_frames.py", args, proj, log_area())
                (st.error if rc else st.success)("실패" if rc else "완료 — 5단계에서 화면을 검토하세요.")
            if vids and (proj / "frames" / "extract_report.json").exists():
                rep = C.rj(proj / "frames" / "extract_report.json")
                st.write({"추출 프레임": rep["n_frames"], "체내 밖 자동표시": rep.get("oob_heuristic")})
        elif vdir:
            st.warning("그 폴더에서 영상을 찾지 못했습니다.")
    with tab2:
        st.markdown("공개 데이터처럼 이미지와 정답이 이미 있는 경우입니다. 파일 경로에서 영상(환자) 번호를 읽을 정규식을 지정합니다. "
                    "공개 데이터 목록은 실습 키트 안내서의 '공개 데이터셋' 표를 보세요.")
        root = st.text_input("데이터 최상위 폴더", "")
        c1, c2 = st.columns(2)
        glob = c1.text_input("이미지 패턴", "images/**/*.png")
        vre = c2.text_input("영상 번호 정규식", r"(video\d+)")
        if pack["task_type"] == "segmentation":
            c3, c4, c5 = st.columns(3)
            mdir = c3.text_input("이미지→마스크 폴더 치환 'a:b'", "images:masks")
            msuf = c4.text_input("마스크 파일 접미사", "")
            colors = c5.text_input("색 마스크면 'class=r,g,b;…'", "")
        else:
            lcsv = st.text_input("라벨 CSV (image,label)", "")
        tag = st.text_input("라벨 출처 이름", "public_dataset")
        if st.button("가져오기") and root:
            args = ["--root", root, "--images", glob, "--video-regex", vre, "--config", proj / "config.json",
                    "--out", proj / "labeled", "--source-tag", tag]
            if pack["task_type"] == "segmentation":
                args += ["--mask-dir", mdir] + (["--mask-suffix", msuf] if msuf else []) + (["--colors", colors] if colors else [])
            else:
                args += ["--labels-csv", lcsv]
            rc, out = C.run_tool("import_folder.py", args, proj, log_area())
            (st.error if rc else st.success)(out[-600:])
    go_next()

# ================================================================== 5. review
elif page == "review":
    need_project()
    st.title("5. 화면 검토 — 체내 밖·식별 가능 장면 제외")
    fm = proj / "frames" / "frames_manifest.csv"
    if not fm.exists():
        st.info("영상에서 추출한 프레임이 없습니다(공개 라벨 데이터만 쓰면 이 단계는 건너뜁니다).")
        go_next()
        st.stop()
    rows = C.read_csv(fm)
    mod = pack["data"].get("modality", "laparoscopic")
    st.markdown("수술실·사람 얼굴·모니터·서류·카메라 삽입 전후 장면을 **모두 체크**하세요. "
                + ("빨간 테두리는 색으로 고른 *후보*일 뿐입니다." if mod in ("laparoscopic", "robotic", "endoscopic")
                   else "이 영상 종류에서는 자동 후보 표시를 쓰지 않으므로 전부 직접 봐야 합니다."))
    vids = sorted({r["video_id"] for r in rows})
    v = st.selectbox("영상", vids)
    vr = [r for r in rows if r["video_id"] == v]
    per = 48
    pages = max(1, (len(vr) - 1) // per + 1)
    pg = st.number_input("페이지", 1, pages, 1) - 1
    excl = set(st.session_state.get("excl", {r["sample_id"] for r in rows if r.get("oob_flag") == "1"}))
    cols = st.columns(8)
    for i, r in enumerate(vr[pg * per:(pg + 1) * per]):
        with cols[i % 8]:
            st.image(str(proj / "frames" / r["image_path"]), use_column_width=True)
            on = st.checkbox(("🔴 " if r.get("oob_score") and float(r["oob_score"] or 0) >= 0.55 else "") + r["sample_id"][-6:],
                             r["sample_id"] in excl, key="x_" + r["sample_id"])
            (excl.add if on else excl.discard)(r["sample_id"])
    st.session_state["excl"] = excl
    reviewer = st.text_input("검토자 이니셜")
    st.caption(f"제외 표시 {len(excl)} / {len(rows)} 프레임. 모든 영상·페이지를 본 뒤 저장하세요.")
    if st.button("검토 완료로 저장", type="primary", disabled=not reviewer):
        for r in rows:
            r["oob_flag"] = "1" if r["sample_id"] in excl else "0"
            r["oob_reviewed"], r["oob_reviewer"] = "1", reviewer
        C.write_csv(rows, fm)
        st.success("저장했습니다.")
    go_next()

# ================================================================== 6. annotate
elif page == "annotate":
    need_project()
    st.title("6. 아노테이션")
    mode, classes = pack["annotation"]["mode"], pack["class_names"]
    with st.expander("라벨 정의서 (이번 연구 확정본)", expanded=True):
        for k, v in pack["label_definitions"].items():
            st.markdown(f"- **{k}**: {v}")
        st.markdown("경계 규칙:\n" + "\n".join(f"- {h}" for h in pack["hard_cases"]))
        st.caption("판단이 안 되면 칠하지/태그하지 말고 판단 보류로 둡니다. 아노테이션 없는 프레임은 학습에서 빠지며, 음성 정답이 아닙니다.")
    fm = proj / "frames" / "frames_manifest.csv"
    annot = st.text_input("판독자 이름(이니셜)", st.session_state.get("annotator", "me"))
    st.session_state["annotator"] = annot
    if mode == "seg":
        t1, t2, t3 = st.tabs(["① CVAT용 배치 만들기", "② CVAT 결과 가져오기", "③ 판독자 간 일치도"])
        with t1:
            if fm.exists() and not all(r.get("oob_reviewed") == "1" for r in C.read_csv(fm)):
                st.warning("5단계 화면 검토를 먼저 끝내세요.")
            elif fm.exists() and st.button("영상별 배치 ZIP 만들기", type="primary"):
                rc, out = C.run_tool("select_frames.py", ["--frames", proj / "frames", "--per-video", pack["annotation"]["frames_per_video"],
                                                          "--min-gap-sec", pack["annotation"]["min_gap_sec"], "--out", proj / "annotate"], proj, log_area())
            for z in sorted((proj / "annotate").glob("cvat_batch_*.zip")) if (proj / "annotate").exists() else []:
                st.download_button(f"⬇ {z.name}", z.read_bytes(), z.name, key=z.name)
            gov = C.rj(proj / "governance_decision.json", {})
            url = "https://app.cvat.ai/" if gov.get("annotation_platform") == "cvat_cloud" else "원내 CVAT 주소"
            st.markdown(f"""
**CVAT에서 할 일** ({url})
1. Tasks → **+** → Create new task. 이름 `{pack['slug']}_<영상번호>`, 라벨 `{', '.join(classes[1:])}` (글자 그대로), 파일은 배치 ZIP 하나.
2. Job을 열고 **Brush** 또는 **Polygon**으로 칠합니다 → **Save mask**(마스크 확정) → 상단 **Save**(작업 저장).
3. 대상이 없는 프레임은 아무것도 칠하지 말고 이름을 적어 두세요(아래 '대상 없음 확인' 칸).
4. **보간·Track·자동 추적은 쓰지 마세요** — 떨어진 프레임이라 아무도 그리지 않은 마스크가 정답처럼 나갑니다. SAM 보조는 손으로 2장 해본 뒤 선택.
5. Actions → **Export task dataset** → 형식 **Segmentation Mask 1.1**, Save images 끔.
[Brush 도구 공식 안내](https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/annotation-with-brush-tool/) · [내보내기](https://docs.cvat.ai/docs/dataset_management/export-datasets/)
""")
        with t2:
            st.markdown("예제로 연습할 때는 **공개 참조 라벨**(`example_reference_labels.zip`)을 먼저 가져온 뒤, **내가 CVAT에서 칠한 export**를 가져옵니다. "
                        "나중에 가져온 라벨이 같은 프레임의 앞 라벨을 덮어씁니다.")
            src_kind = st.radio("이 파일의 출처", ["내 라벨", "공개 참조 라벨", "다른 판독자"], horizontal=True)
            src_tag = {"내 라벨": annot, "공개 참조 라벨": "public_reference"}.get(src_kind) or st.text_input("판독자 이름", "rater2")
            use_main = st.checkbox("학습 데이터에 반영", src_kind != "다른 판독자",
                                   help="다른 판독자 라벨은 보통 일치도 계산에만 쓰고 학습 데이터에는 넣지 않습니다.")
            ups = st.file_uploader("CVAT Segmentation Mask 1.1 export ZIP(들)", type="zip", accept_multiple_files=True)
            empty = st.text_area("대상 없음 확인 프레임 이름(한 줄에 하나)", "")
            if st.button("가져오기", type="primary") and ups:
                (proj / "exports").mkdir(exist_ok=True)
                rex = proj / "exports" / "reviewed_empty.txt"
                rex.write_text(empty)
                lab = proj / "labeled" / "labeled_manifest.csv"
                for u in ups:
                    dst = proj / "exports" / f"{src_tag}_{u.name}"
                    dst.write_bytes(u.getbuffer())
                    targets = [proj / "by_source" / src_tag] + ([proj / "labeled"] if use_main else [])
                    for tgt in targets:
                        cur_m = tgt / "labeled_manifest.csv"
                        base = cur_m if cur_m.exists() else fm
                        args = ["seg", "--manifest", base, "--export", dst, "--config", proj / "config.json", "--out", tgt,
                                "--source-tag", src_tag, "--reviewed-empty", rex]
                        if base == fm:
                            args += ["--frames", proj / "frames"]
                        rc, out = C.run_tool("cvat_import.py", args, proj, log_area())
                        if rc:
                            st.error(out[-600:])
                            st.stop()
                st.success(C.rj(proj / "labeled" / "import_report.json") if use_main else f"{src_tag} 라벨 저장(일치도용)")
        with t3:
            st.markdown("같은 프레임을 서로 다른 출처가 칠했을 때 얼마나 겹치는지(Dice) 봅니다. 예: **내 라벨 vs 공개 참조 라벨**, 판독자 1 vs 2. "
                        "낮은 프레임은 라벨 정의를 다듬을 지점입니다.")
            srcs = sorted(d.name for d in (proj / "by_source").iterdir()) if (proj / "by_source").exists() else []
            if len(srcs) < 2:
                st.info("출처가 2개 이상 있어야 합니다(②에서 각각 가져오기).")
            else:
                a_ = st.selectbox("기준(A)", srcs, index=srcs.index("public_reference") if "public_reference" in srcs else 0)
                b_ = st.selectbox("비교(B)", [x for x in srcs if x != a_])
                if st.button("일치도 계산"):
                    rc, out = C.run_tool("compare_labels.py", ["--a", proj / "by_source" / a_ / "labeled_manifest.csv",
                                                                "--b", proj / "by_source" / b_ / "labeled_manifest.csv", "--config", proj / "config.json",
                                                                "--out", proj / "agreement" / f"{a_}_vs_{b_}", "--name-a", a_, "--name-b", b_], proj)
                    (st.error if rc else st.json)(out[-600:] if rc else C.rj(proj / "agreement" / f"{a_}_vs_{b_}" / "agreement_summary.json"))
    else:
        if not fm.exists():
            st.info("4단계에서 프레임을 추출하세요.")
            st.stop()
        rows = [r for r in C.read_csv(fm) if r.get("oob_flag") != "1"]
        tags_p = proj / "labels" / f"tags_{annot}.csv"
        tags = {r["sample_id"]: r["label"] for r in C.read_csv(tags_p)} if tags_p.exists() else {}
        if mode == "tag":
            # pick the same spread of frames select_frames would use, per video
            per = int(pack["annotation"]["frames_per_video"])
            todo = []
            for v in sorted({r["video_id"] for r in rows}):
                vr = [r for r in rows if r["video_id"] == v]
                step = max(1, len(vr) // per)
                todo += vr[::step][:per]
            remaining = [r for r in todo if r["sample_id"] not in tags]
            st.progress(1 - len(remaining) / max(len(todo), 1), text=f"{len(todo) - len(remaining)} / {len(todo)} 프레임")
            idx = st.session_state.get("tag_i", 0)
            if remaining:
                r = remaining[min(idx, len(remaining) - 1)]
                c1, c2 = st.columns([3, 1])
                c1.image(str(proj / "frames" / r["image_path"]), caption=r["sample_id"], use_column_width=True)
                with c2:
                    st.markdown("**이 프레임은?**")
                    def _tag(sid, lab):
                        tags[sid] = lab
                        C.write_csv([{"sample_id": k, "label": v} for k, v in tags.items()], tags_p)
                    for c in classes:
                        st.button(c, key="b_" + c, use_container_width=True, on_click=_tag, args=(r["sample_id"], c))
                    st.button("판단 보류", use_container_width=True, on_click=_tag, args=(r["sample_id"], "__uncertain__"))
            else:
                st.success("선택된 프레임을 모두 태그했습니다.")
        else:
            st.markdown("영상을 보며 phase가 바뀌는 시각(초)을 적습니다. 애매한 전환 구간은 비워 두세요. 구간이 겹치면 그 시간 프레임은 빠집니다.")
            ivp = proj / "labels" / "phases.csv"
            cur = C.read_csv(ivp) if ivp.exists() else [{"video_id": v, "start_sec": 0, "end_sec": 0, "label": classes[0]}
                                                         for v in sorted({r["video_id"] for r in rows})]
            ed = st.data_editor(cur, num_rows="dynamic", use_container_width=True,
                                column_config={"label": st.column_config.SelectboxColumn(options=classes)})
            if st.button("구간 저장·적용", type="primary"):
                C.write_csv(list(ed), ivp)
                rc, out = C.run_tool("cvat_import.py", ["interval", "--manifest", fm, "--intervals", ivp, "--frames", proj / "frames",
                                                         "--config", proj / "config.json", "--out", proj / "labeled", "--source-tag", annot], proj, log_area())
        if mode == "tag" and st.button("태그를 학습 데이터로 적용", type="primary"):
            st.success(C.apply_tags(proj, annot, classes))
    go_next()

# ================================================================== 7. split
elif page == "split":
    need_project()
    st.title("7. 데이터 구성 — 환자 단위 분할과 점검")
    lab = proj / "labeled" / "labeled_manifest.csv"
    if not lab.exists():
        st.info("6단계 아노테이션을 먼저 적용하세요.")
        st.stop()
    rows = C.read_csv(lab)
    pats = sorted({r["patient_id"] for r in rows if r.get("annotation_status") == "done"})
    st.markdown(f"아노테이션된 환자 **{len(pats)}명**. 같은 환자는 한 집합에만 들어갑니다.")
    c1, c2 = st.columns(2)
    ratios = c1.text_input("학습,검증,시험 비율", pack["split"]["ratios"])
    how = c2.radio("시험군", ["무작위", "고정(시간적/특정 환자)"], horizontal=True)
    fixed = st.multiselect("시험 환자(나중 수술일 순 마지막 환자 등)", pats, pack["split"].get("test_patients", [])) if how != "무작위" else []
    if st.button("분할하고 점검", type="primary"):
        pack["split"].update({"ratios": ratios, "test_patients": fixed})
        C.wj(proj / "pack.json", pack)
        args = ["--manifest", lab, "--ratios", ratios, "--seed", pack["split"]["seed"], "--out", proj / "split"]
        if fixed:
            args += ["--test-patients", ",".join(fixed)]
        rc, out = C.run_tool("make_splits.py", args, proj, log_area())
        if not rc:
            C.run_tool("data_audit.py", ["--manifest", proj / "split" / "split_manifest.csv", "--config", proj / "config.json", "--out", proj / "audit"], proj)
        else:
            st.error(out[-600:])
    s = C.rj(proj / "split" / "split_summary.json")
    if s:
        st.json({k: s[k] for k in ("n_patients", "n_frames", "leakage", "warnings")})
        show_md(proj / "audit" / "data_audit.md")
    go_next()

# ================================================================== 8. search
elif page == "search":
    need_project()
    st.title("8. 모델 후보 비교 (검증 자료만 사용)")
    if not (proj / "split" / "split_manifest.csv").exists():
        st.info("7단계 분할을 먼저 하세요.")
        st.stop()
    from models import CLS_ARCHS, SEG_ARCHS
    zoo = SEG_ARCHS if pack["task_type"] == "segmentation" else CLS_ARCHS
    st.markdown("같은 조건으로 짧게 학습해 **검증 점수**로 비교합니다. 시험 자료는 열지 않습니다. 점수 차이가 작으면 더 가벼운 모델을 고릅니다.")
    st.table([{"모델": a, "설명": d} for a, d in zoo.items()])
    cands = st.multiselect("후보", list(zoo), [c for c in pack["metrics"].get("candidates", list(zoo)[:3]) if c in zoo])
    c1, c2, c3 = st.columns(3)
    ep = c1.number_input("후보당 epoch", 1, 50, 4)
    folds = c2.number_input("교차검증 fold (1=고정 검증셋)", 1, 10, 1)
    tol = c3.number_input("동점 허용폭", 0.0, 0.1, 0.01)
    if st.button("비교 시작", type="primary") and cands:
        with st.status("후보 학습 중… (후보 수 × fold × epoch 만큼 걸립니다)", expanded=True):
            rc, out = C.run_tool("model_search.py", ["--config", proj / "config.json", "--manifest", proj / "split" / "split_manifest.csv",
                                                      "--out", proj / "search", "--candidates", ",".join(cands), "--epochs", ep,
                                                      "--folds", folds, "--tolerance", tol], proj, log_area())
    s = C.rj(proj / "search" / "search_summary.json")
    if s:
        st.dataframe([{k: b[k] for k in ("arch", "params_M", "val_mean", "val_scores", "error")} for b in s["leaderboard"]], use_container_width=True)
        st.success(f"선택: **{s['chosen']}** — {s['chosen_reason']}")
        if st.button("이 모델을 학습 설정에 반영"):
            pack["training"]["arch"] = s["chosen"]
            C.wj(proj / "pack.json", pack)
            C.write_config(proj, pack)
            st.success("반영했습니다.")
    go_next()

# ================================================================== 9. train
elif page == "train":
    need_project()
    st.title("9. 학습·평가")
    if not (proj / "split" / "split_manifest.csv").exists():
        st.info("7단계 분할을 먼저 하세요.")
        st.stop()
    tr = pack["training"]
    c1, c2, c3 = st.columns(3)
    c1.write(f"모델: **{tr['arch']}**")
    ep = c2.number_input("epoch", 1, 200, int(tr["epochs"]))
    pre = c3.checkbox("ImageNet 사전학습 사용(인터넷 필요)", True)
    st.caption("가장 좋은 epoch는 검증으로 고르고, 그 뒤에 시험 자료로 **한 번** 평가합니다. 결과를 보고 설정을 바꿔 다시 돌렸다면 원고에 적어야 합니다.")
    if st.button("학습 시작", type="primary"):
        pack["training"]["epochs"] = int(ep)
        C.wj(proj / "pack.json", pack)
        C.write_config(proj, pack)
        out_dir = proj / "runs" / C.dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args = ["--config", proj / "config.json", "--manifest", proj / "split" / "split_manifest.csv", "--out", out_dir]
        if not pre:
            args.append("--no-pretrained")
        with st.status("학습 중…", expanded=True):
            rc, out = C.run_tool("train.py", args, proj, log_area())
        if rc == 0:
            C.run_tool("make_report.py", ["--split", proj / "split" / "split_summary.json", "--run", out_dir, "--out", out_dir / "results"], proj)
        else:
            st.error(out[-800:])
    run = C.latest_run(proj)
    if run:
        m = C.rj(run / "metrics.json")
        st.subheader(f"최근 결과 · {run.name}")
        cc = st.columns(4)
        cc[0].metric("선택 epoch", m["selected_epoch"])
        cc[1].metric("검증 점수 (전→후)", f"{m['val_selected']:.3f}", f"{m['val_selected'] - m['val_before_training']:+.3f}")
        cc[2].metric("가중치 변경", "✔" if m["weights_changed"] else "✘")
        cc[3].metric("재로딩 검증", "✔" if m["checkpoint_reload_verified"] else "✘")
        show_md(run / "results" / "table2_performance.md")
        g1, g2 = st.columns(2)
        for g, f in ((g1, "fig_learning_curve.png"), (g2, "fig_per_case.png")):
            if (run / "results" / f).exists():
                g.image(str(run / "results" / f))
        ex = sorted((run / "examples").glob("*.png")) if (run / "examples").exists() else []
        if ex:
            st.markdown("**예시 (원본 | 정답 | 예측)** — worst부터 보고 오류 유형을 적어 두세요.")
            for e in ex:
                st.image(str(e), caption=e.name)
        with st.expander("영상에 적용해 보기"):
            vp = st.text_input("영상 파일 경로(학습·검증에 안 쓴 영상)")
            if st.button("추론") and vp:
                C.run_tool("infer_video.py", ["--checkpoint", run / "checkpoint_best.pt", "--video", vp, "--fps", 2, "--out", run / "video"], proj, log_area())
            if (run / "video" / "overlay.mp4").exists():
                st.video(str(run / "video" / "overlay.mp4"))
            if (run / "video" / "timeline.png").exists():
                st.image(str(run / "video" / "timeline.png"))
        with st.expander("학습곡선 — 환자를 더 모으면 얼마나 좋아질까?"):
            st.markdown("학습 환자 수를 늘려 가며 다시 학습해 검증 점수 변화를 보고, 곡선을 그려 목표 점수에 필요한 환자 수를 추정합니다. 시험 자료는 쓰지 않습니다.")
            tgt = st.number_input("목표 검증 점수(비우면 정체치의 95%)", 0.0, 1.0, 0.0)
            reps = st.number_input("크기별 반복", 1, 5, 2)
            if st.button("학습곡선 실행"):
                args = ["curve", "--config", proj / "config.json", "--manifest", proj / "split" / "split_manifest.csv",
                        "--out", proj / "sample_size", "--reps", reps, "--epochs", max(2, int(ep) // 2)]
                if tgt:
                    args += ["--target", tgt]
                with st.status("학습곡선 계산 중…", expanded=True):
                    C.run_tool("sample_size.py", args, proj, log_area())
            lc = C.rj(proj / "sample_size" / "sample_size_curve.json")
            if lc:
                if (proj / "sample_size" / "learning_curve.png").exists():
                    st.image(str(proj / "sample_size" / "learning_curve.png"))
                st.json(lc.get("fit") or {"fit_error": lc.get("fit_error")})
        with st.expander("GPU가 필요하면 — Colab용 묶음 만들기"):
            gov = C.rj(proj / "governance_decision.json", {})
            if gov.get("compute_platform") != "cloud_approved":
                st.warning("거버넌스 결정이 local_only입니다. 외부 클라우드로 내보내지 마세요.")
            elif st.button("study_bundle.zip + 노트북 만들기"):
                C.run_tool("bundle.py", ["--config", proj / "config.json", "--split", proj / "split", "--governance",
                                         proj / "governance_decision.json", "--out", proj / "colab" / "study_bundle.zip"], proj)
                C.run_tool("build_notebook.py", ["--config", proj / "config.json", "--title", pack["title_ko"],
                                                 "--out", proj / "colab" / "train_eval.ipynb"], proj)
            for f in ("study_bundle.zip", "train_eval.ipynb"):
                if (proj / "colab" / f).exists():
                    st.download_button(f"⬇ {f}", (proj / "colab" / f).read_bytes(), f, key="dl_" + f)
    go_next()

# ================================================================== 10. external
elif page == "external":
    need_project()
    st.title("10. 외부검증")
    run = C.latest_run(proj)
    if not run:
        st.info("9단계 학습을 먼저 하세요.")
        st.stop()
    st.markdown("""
**외부검증**은 학습·모델선택에 전혀 쓰지 않은 **다른 기관·다른 시기·다른 장비** 환자에서 *고정된* 모델을 평가하는 것입니다. 재학습·임계값 조정은 하지 않습니다.
외부 데이터는 다른 SurgVision 프로젝트(같은 클래스 이름)로 준비하거나, 4단계 B처럼 라벨 폴더를 가져온 프로젝트를 쓰면 됩니다.
""")
    others = [p for p in C.list_projects() if p != slug and (C.pdir(p) / "labeled" / "labeled_manifest.csv").exists()]
    src = st.selectbox("외부 데이터 프로젝트", others) if others else None
    name = st.text_input("외부 세트 이름", "external_hospitalB")
    if src and st.button("외부검증 실행", type="primary"):
        op = C.rj(C.pdir(src) / "pack.json")
        ext_rows = C.read_csv(C.pdir(src) / "labeled" / "labeled_manifest.csv")
        int_rows = C.read_csv(proj / "split" / "split_manifest.csv")
        same_img = {Path(r["image_path"]).resolve() for r in ext_rows} & {Path(r["image_path"]).resolve() for r in int_rows if r.get("annotation_status") == "done"}
        if op["class_names"] != pack["class_names"]:
            st.error(f"클래스 이름이 다릅니다: {op['class_names']} vs {pack['class_names']}")
        elif same_img:
            st.error(f"외부 세트의 이미지 {len(same_img)}장이 이 프로젝트의 학습/검증/시험에도 있습니다. 독립된 외부 데이터가 아니므로 외부검증이라고 할 수 없습니다.")
        else:
            rc, out = C.run_tool("evaluate.py", ["--checkpoint", run / "checkpoint_best.pt", "--manifest", C.pdir(src) / "labeled" / "labeled_manifest.csv",
                                                  "--name", name, "--out", proj / "external" / name, "--internal", run / "metrics.json"], proj, log_area())
    for d in sorted((proj / "external").glob("*/")) if (proj / "external").exists() else []:
        st.subheader(d.name)
        show_md(d / "table3_external.md")
    go_next()

# ================================================================== 11. paper
elif page == "paper":
    need_project()
    st.title("11. 결과·원고 초안")
    run = C.latest_run(proj)
    if not run:
        st.info("9단계 학습을 먼저 하세요.")
        st.stop()
    if st.button("원고 초안 만들기", type="primary"):
        args = ["--project", proj, "--run", run, "--out", proj / "manuscript_draft.md", "--audit", proj / "audit"]
        for d in sorted((proj / "external").glob("*/")) if (proj / "external").exists() else []:
            args += ["--external", d]
        if (proj / "search" / "search_summary.json").exists():
            args += ["--search", proj / "search"]
        if (proj / "sample_size" / "sample_size_precision.json").exists():
            args += ["--sample-size", proj / "sample_size"]
        rc, out = C.run_tool("draft_manuscript.py", args, proj)
        (st.error if rc else st.success)(out[-400:])
    if (proj / "manuscript_draft.md").exists():
        txt = (proj / "manuscript_draft.md").read_text(encoding="utf-8")
        st.download_button("⬇ manuscript_draft.md", txt, "manuscript_draft.md")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for sub in ("pack.json", "config.json", "governance_decision.json", "manuscript_draft.md"):
                if (proj / sub).exists():
                    z.write(proj / sub, sub)
            for folder in ("split", "audit", "search", "sample_size", "external"):
                for f in (proj / folder).rglob("*") if (proj / folder).exists() else []:
                    if f.is_file() and f.suffix in (".json", ".md", ".csv", ".png"):
                        z.write(f, f.relative_to(proj))
            for f in run.rglob("*"):
                if f.is_file() and f.suffix in (".json", ".md", ".csv", ".png", ".mp4"):
                    z.write(f, f.relative_to(proj))
        st.download_button("⬇ 결과 묶음(.zip, 원본 영상·대응표 제외)", buf.getvalue(), f"{slug}_results.zip")
        st.markdown(txt)
        st.info("다음: Introduction·Discussion은 PubMed로 확인한 문헌으로 직접 씁니다. Claude Code에서 'surgvision-loop 결과로 원고 보완'을 요청하거나 "
                "Aside S6 프롬프트를 쓰면 숫자 출처를 지키며 문장을 다듬을 수 있습니다.")
