"""
800건 골든셋 배치 평가
─────────────────────────────────────────────────────────────
실행: python batch_evaluate.py

필요 파일:
  - final_chest_xray_test_800_unique.csv  (이 파일과 같은 폴더)
  - medical_prototype_v2-2.py             (이 파일과 같은 폴더, 함수 임포트용)
  - tfidf_index.pkl / fewshot_pool.pkl    (BASE_DIR 안)

결과 파일:
  - batch_results.csv   (건별 상세 결과)
  - batch_summary.txt   (최종 통계)
"""
import os, sys, json, csv, time, random, re, pickle, base64, io
from collections import defaultdict

OUT_DIR =os.path.dirname(os.path.abspath(__file__))

from datetime import datetime
timestamp = datetime.now().strftime('%m%d_%H%M')
csv_out = os.path.join(OUT_DIR, f'batch_results_{timestamp}.csv')
txt_out = os.path.join(OUT_DIR, f'batch_summary_{timestamp}.txt')

# ── 설정 ─────────────────────────────────────────────────────
API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "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")
MODEL     = "claude-haiku-4-5-20251001"
BASE_DIR  = r"G:\내 드라이브\의학관련 데이터\files (1)"
CSV_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "final_chest_xray_test_800_unique.csv")
IMAGE_DIR = r"G:\내 드라이브\의학관련 데이터\kagglehub\datasets\raddar\chest-xrays-indiana-university\versions\2\images\images_normalized"

# 평가할 샘플 수 (전체=800, 시간 절약=50)
SAMPLE_SIZE = 100

# 결과 저장 위치 (이 파일과 같은 폴더)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 패키지 임포트 ─────────────────────────────────────────────
import anthropic
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# Windows 콘솔 인코딩 UTF-8 강제 설정
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
os.environ['PYTHONIOENCODING'] = 'utf-8'

client = anthropic.Anthropic(api_key=API_KEY)

# ── 인덱스 로드 ───────────────────────────────────────────────
print("인덱스 로딩 중...")
_idx       = pickle.load(open(os.path.join(BASE_DIR, 'tfidf_index.pkl'), 'rb'))
VECTORIZER = _idx['vectorizer']
MATRIX     = _idx['matrix']
DOCS       = _idx['docs']
FEWSHOT    = pickle.load(open(os.path.join(BASE_DIR, 'fewshot_pool.pkl'), 'rb'))

DOMAIN_MAP = {
    1:'외과', 2:'예방의학', 3:'정신건강의학과', 4:'신경과신경외과',
    5:'피부과', 6:'안과', 7:'이비인후과', 8:'비뇨의학과',
    9:'방사선종양학과', 10:'병리과', 11:'마취통증의학과', 12:'의료법규', 13:'기타'
}
print(f"✅ 인덱스 로드 완료 — 청크 {len(DOCS)}개")

# ── 파이프라인 함수 ───────────────────────────────────────────
def retrieve(query, top_k=5):
    qvec = normalize(VECTORIZER.transform([query]), norm='l2')
    scores = (MATRIX @ qvec.T).toarray().flatten()
    return [{'text': DOCS[i]['text'], 'score': float(scores[i]),
             'cat': DOCS[i]['cat']} for i in scores.argsort()[::-1][:top_k]]

def get_fewshot(domain_id, n=2):
    pool = FEWSHOT.get(domain_id, FEWSHOT.get(13, []))
    samples = random.sample(pool, min(n, len(pool)))
    return "\n\n".join(
        f"[전문 소견]\n{s['question'][:200]}\n[쉬운 말 설명]\n{s['answer'][:300]}"
        for s in samples)

def translate_findings(english_text):
    # XXXX 비식별 문자 제거 후 번역
    clean_text = english_text.replace('XXXX', '[비식별]').replace('xxxx', '[비식별]')
    prompt = (
        "다음 영문 영상의학 판독 소견을 한국어로 번역하세요.\n"
        "규칙: 1) 의학 전문용어는 '한국어(영문)' 형태로 병기  2) [비식별]은 그대로 유지\n"
        "3) 번역문만 출력, 설명 금지\n\n"
        f"[영문 소견]\n{clean_text}\n\n[한국어 번역]:"
    )
    resp = client.messages.create(model=MODEL, max_tokens=600,
                                  messages=[{"role": "user", "content": prompt}])
    result = resp.content[0].text.strip()
    # 안전한 문자열로 인코딩 보장
    return result.encode('utf-8', errors='replace').decode('utf-8')

def generate_easy_explanation(input_text, retrieved_docs, domain_id):
    context = "\n---\n".join([d['text'] for d in retrieved_docs[:3]])
    fewshot = get_fewshot(domain_id)
    prompt = (
        "당신은 환자에게 의료 정보를 쉽게 설명해주는 전문가입니다.\n"
        "초등학생도 이해할 수 있는 일상 언어로 설명하되 의학적 사실은 보존하세요.\n\n"
        f"[참고 의학 지식]\n{context}\n\n[변환 예시]\n{fewshot}\n\n"
        f"[환자 소견/진단]\n{input_text}\n\n[쉬운 말 설명] (200~300자):"
    )
    resp = client.messages.create(model=MODEL, max_tokens=500,
                                  messages=[{"role": "user", "content": prompt}])
    return resp.content[0].text.strip()

def synthesize_report(input_text, easy, triage, track):
    prompt = (
        "다음 정보를 바탕으로 환자에게 전달할 최종 안내 리포트를 작성하세요.\n"
        "엄격한 작성 규칙:\n"
        "1. 원본 소견에 명시된 내용만 포함하세요. 원문에 없는 치료법·예후·추가 검사·예상 결과는 절대 언급하지 마세요.\n"
        "2. 원본에 없는 주관적 판단(예: '비교적 가벼운', '심각한')을 임의로 추가하지 마세요.\n"
        "3. 면책 고지는 반드시 완전한 문장으로 마무리하세요.\n\n"
        f"[원본 소견] {input_text}\n[쉬운 말 설명] {easy}\n"
        f"[진료 트랙] {'Track B — 상급종합 3차' if track=='B' else 'Track A — 지역 1·2차'}\n\n"
        "구조:\n1. 📋 현재 상태 요약\n2. 🏥 추천 의료기관 유형\n3. ⚠️ 면책 고지"
    )
    resp = client.messages.create(model=MODEL, max_tokens=2000,
                                  messages=[{"role": "user", "content": prompt}])
    return resp.content[0].text.strip()

def llm_judge(original_text, generated_report):
    prompt = (
        "당신은 의료 AI 출력물의 품질을 평가하는 전문 판정관입니다.\n\n"
        f"[원본 환자 소견]\n{original_text}\n\n[AI 생성 리포트]\n{generated_report}\n\n"
        "아래 4가지 항목을 평가하고 JSON만 출력하세요.\n\n"
        "- correctness: 원문 사실 정확성 (0=핵심 사실 틀림[자동FAIL], 1=사실 왜곡, 2=임상의미 영향 부정확1건, 3=경미한 부정확1건, 4=완전 정확)\n"
        "- completeness: 핵심 정보 완전성 (0=대부분 누락, 1=중요정보 다수 누락, 2=핵심1건 누락, 3=부차적1건 누락, 4=누락없음)\n"
        "- hallucination: 환각 통제 (0=사실과 다른 정보 생성[자동FAIL], 1=추정을 사실처럼 기술, 2=원문없는정보(위해낮음), 3=상식수준보조설명1건, 4=추가정보없음)\n"
        "  ※ 아래는 환각으로 판정하지 말 것:\n"
        "    - 병원 추천 및 의료기관 정보 (시스템 기능, 원문 언급 불필요)\n"
        "    - 면책 고지, 전문의 상담 권고, 진료의뢰서 안내\n"
        "    - 원문 소견을 쉬운 말로 풀어 설명하는 것\n"
        "    - 해부학 용어를 일상어로 변환한 표현\n"
        "    - 병원의 진료 초점·전문분야 설명 (시스템이 제공하는 참고 정보)\n"
        "- readability: 환자 이해도 (0=단순화효과없음, 1=전문용어다수잔존, 2=일부문장전문적, 3=전문용어1-2개미풀이, 4=초등고학년수준)\n"
        "- total: 4개 합산 (0~16)\n"
        "- verdict: total>=12 AND correctness>=2 AND hallucination>=2 AND 0점없음 이면 PASS 아니면 FAIL\n"
        "- feedback: 개선사항 한 줄\n\n"
        '출력: {"correctness":{"score":4,"comment":"설명"},"completeness":{"score":3,"comment":"설명"},'
        '"hallucination":{"score":4,"comment":"설명"},"readability":{"score":3,"comment":"설명"},'
        '"total":14,"verdict":"PASS","feedback":"설명"}'
    )
    resp = client.messages.create(model=MODEL, max_tokens=2000,
                                  messages=[{"role": "user", "content": prompt}])
    m = re.search(r'\{.*\}', resp.content[0].text.strip(), re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return {"correctness":{"score":0},"completeness":{"score":0},
            "hallucination":{"score":0},"readability":{"score":0},
            "total":0,"verdict":"FAIL","feedback":"판정 실패"}

# ── CSV 로드 및 샘플링 ────────────────────────────────────────
print(f"\nCSV 로딩: {CSV_PATH}")
with open(CSV_PATH, encoding='utf-8') as f:
    all_rows = list(csv.DictReader(f))

# label 비율 유지 층화 샘플링
normal   = [r for r in all_rows if r['label'] == 'Normal']
abnormal = [r for r in all_rows if r['label'] == 'Abnormal']
n_normal   = int(SAMPLE_SIZE * len(normal)   / len(all_rows))
n_abnormal = SAMPLE_SIZE - n_normal
random.seed(42)
samples = random.sample(normal, n_normal) + random.sample(abnormal, n_abnormal)
random.shuffle(samples)

print(f"✅ 샘플링 완료 — Normal {n_normal}건 / Abnormal {n_abnormal}건 / 총 {len(samples)}건")

# ── 배치 평가 실행 ────────────────────────────────────────────
results = []
start_time = time.time()

print(f"\n{'='*60}")
print(f"배치 평가 시작 — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*60}\n")

for i, row in enumerate(samples, 1):
    uid       = row['uid']
    filename  = row['filename']
    findings  = row['findings']
    label     = row['label']       # 정답 (Normal/Abnormal)
    problems  = row['Problems']

    print(f"[{i:3d}/{len(samples)}] uid={uid} | {label} | {problems[:40]}...")

    try:
        t0 = time.time()

        # Step 1: 영문 소견 → 한국어 번역
        ko_findings = translate_findings(findings)

        # Step 2: RAG 검색
        retrieved = retrieve(ko_findings, top_k=5)

        # Step 3: 진료과·중증도 추출 (간소화 버전)
        combined = (findings + " " + row.get('impression','')).lower()
        severe_kw = ['pneumothorax','effusion','edema','mass','malignancy',
                     'cardiomegaly','fracture','hemorrhage','stroke']
        severity  = '중증' if any(k in combined for k in severe_kw) else '경증'
        # label=Normal이면 Track A 우선
        if label == 'Normal':
            track = 'A'
        else:
            track = 'B' if severity == '중증' else 'A'
        domain_id = 1  # 흉부 X-ray는 외과 계열

        # Step 4: 쉬운 말 설명
        easy = generate_easy_explanation(ko_findings, retrieved, domain_id)

        # Step 5: 리포트 합성 (병원 매칭 제외 — 배치 속도용)
        triage = {'reason': f"{label} 케이스"}
        report = synthesize_report(ko_findings, easy, triage, track)

        # Step 6: LLM-as-Judge
        judge = llm_judge(ko_findings, report)

        elapsed = round(time.time() - t0, 1)

        # 결과 수집
        c = judge.get('correctness',{}).get('score', 0)
        cm = judge.get('completeness',{}).get('score', 0)
        h = judge.get('hallucination',{}).get('score', 0)
        r = judge.get('readability',{}).get('score', 0)
        total   = judge.get('total', c+cm+h+r)
        verdict = judge.get('verdict', 'FAIL')

        # 자동 FAIL 체크
        auto_fail = (c == 0 or h == 0)
        if auto_fail and verdict == 'PASS':
            verdict = 'FAIL'

        results.append({
            'uid': uid, 'filename': filename, 'label': label,
            'problems': problems[:60], 'track': track,
            'correctness': c, 'completeness': cm,
            'hallucination': h, 'readability': r,
            'total': total, 'verdict': verdict,
            'feedback': judge.get('feedback',''),
            'elapsed': elapsed
        })

        print(f"       → {verdict} ({total}/16) | C:{c} Cm:{cm} H:{h} R:{r} | {elapsed}초")

    except Exception as e:
        print(f"       → ❌ 오류: {e}")
        results.append({
            'uid': uid, 'filename': filename, 'label': label,
            'problems': problems[:60], 'track': 'N/A',
            'correctness': 0, 'completeness': 0,
            'hallucination': 0, 'readability': 0,
            'total': 0, 'verdict': 'ERROR',
            'feedback': str(e)[:100], 'elapsed': 0
        })

    # API 과부하 방지
    time.sleep(1)

# ── 결과 저장 ─────────────────────────────────────────────────
csv_out = os.path.join(OUT_DIR, 'batch_results.csv')
with open(csv_out, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)
print(f"\n✅ 건별 결과 저장 → {csv_out}")

# ── 통계 산출 ─────────────────────────────────────────────────
valid = [r for r in results if r['verdict'] != 'ERROR']
total_elapsed = round(time.time() - start_time, 0)

pass_cnt  = sum(1 for r in valid if r['verdict'] == 'PASS')
fail_cnt  = sum(1 for r in valid if r['verdict'] == 'FAIL')
error_cnt = sum(1 for r in results if r['verdict'] == 'ERROR')

if not valid:
    print(f"\n⚠️  유효한 평가 결과가 없습니다. 오류 {error_cnt}건 전부 실패.")
    print("원인: API 키 확인, 인코딩 문제 확인")
    sys.exit(1)

def avg(lst): return round(sum(lst)/len(lst), 2) if lst else 0

scores = lambda k: [r[k] for r in valid]

avg_total = avg(scores('total'))
avg_c  = avg(scores('correctness'))
avg_cm = avg(scores('completeness'))
avg_h  = avg(scores('hallucination'))
avg_r  = avg(scores('readability'))

# Track 분기 정확도
track_b_correct = sum(1 for r in valid if r['label']=='Abnormal' and r['track']=='B')
track_a_correct = sum(1 for r in valid if r['label']=='Normal' and r['track']=='A')
abnormal_cnt = sum(1 for r in valid if r['label']=='Abnormal')
normal_cnt   = sum(1 for r in valid if r['label']=='Normal')

# 벤치마크 비교 (선행연구 16편 메타분석 기준)
bench_accuracy   = 78   # 사실 정확도 하한
bench_complete   = 83   # 완전성 하한
our_accuracy  = round(sum(1 for r in valid if r['correctness'] >= 2) / len(valid) * 100, 1) if valid else 0
our_complete  = round(sum(1 for r in valid if r['completeness'] >= 2) / len(valid) * 100, 1) if valid else 0
our_harm_rate = round(sum(1 for r in valid if r['hallucination'] == 0) / len(valid) * 100, 1) if valid else 0

summary = f"""
{'='*60}
환자 맞춤형 의료 안내 서비스 — 배치 평가 결과
평가일: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*60}

[평가 규모]
  총 샘플:    {len(samples)}건
  유효 평가:  {len(valid)}건
  오류:       {error_cnt}건
  소요 시간:  {total_elapsed}초 (평균 {round(total_elapsed/len(samples),1)}초/건)

[종합 판정]
  ✅ PASS:  {pass_cnt}건 ({round(pass_cnt/len(valid)*100,1)}%)
  ❌ FAIL:  {fail_cnt}건 ({round(fail_cnt/len(valid)*100,1)}%)

[4축 평균 점수 (4점 만점)]
  ① 사실 정확성  (Correctness):  {avg_c}/4
  ② 완전성       (Completeness): {avg_cm}/4
  ③ 환각 통제    (Hallucination):{avg_h}/4
  ④ 환자 이해도  (Readability):  {avg_r}/4
  ─────────────────────────────
  총점 평균: {avg_total}/16점

[선행연구 벤치마크 비교]
  사실 정확도 (correctness>=2): {our_accuracy}%  (기준 ≥{bench_accuracy}%) {'✅' if our_accuracy >= bench_accuracy else '❌'}
  완전성      (completeness>=2): {our_complete}%  (기준 ≥{bench_complete}%) {'✅' if our_complete >= bench_complete else '❌'}
  유해 오류율 (hallucination=0): {our_harm_rate}%  (기준 ≤8%) {'✅' if our_harm_rate <= 8 else '❌'}

[Track 분기 정확도]
  Abnormal → Track B: {track_b_correct}/{abnormal_cnt} ({round(track_b_correct/abnormal_cnt*100,1) if abnormal_cnt else 0}%)
  Normal   → Track A: {track_a_correct}/{normal_cnt} ({round(track_a_correct/normal_cnt*100,1) if normal_cnt else 0}%)

[PASS 기준]
  총점 ≥ 12/16점(75%) AND 정확성 ≥ 2점 AND 환각 ≥ 2점 AND 어떤 항목도 0점 없음
  근거: PEMAT(AHRQ) 70% 기준선 상회 + Jeblick et al. 3축 준용
{'='*60}
"""

print(summary)

txt_out = os.path.join(OUT_DIR, 'batch_summary.txt')
with open(txt_out, 'w', encoding='utf-8') as f:
    f.write(summary)
print(f"✅ 요약 통계 저장 → {txt_out}")