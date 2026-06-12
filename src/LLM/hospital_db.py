"""
권역별 병원 더미 DB — 실제 서비스에서는 심평원 API로 교체
"""

REGIONS = {
    '수도권': ['서울','경기','인천'],
    '충청권': ['대전','충남','충북','세종'],
    '호남권': ['광주','전남','전북'],
    '대구경북권': ['대구','경북'],
    '부산경남권': ['부산','울산','경남'],
    '강원권': ['강원'],
    '제주권': ['제주'],
}

# Track A: 1·2차 지역 병원 (권역별)
TRACK_A_DB = {
    '수도권': [
        {'name':'서울 열린내과의원','addr':'서울 강남구','depts':['내과','외과','신경과'],'type':'의원','rating':4.5},
        {'name':'경기중앙병원','addr':'경기 수원시','depts':['정형외과','내과','피부과'],'type':'병원','rating':4.3},
        {'name':'인천현대의원','addr':'인천 남동구','depts':['내과','이비인후과','안과'],'type':'의원','rating':4.2},
        {'name':'강남연세병원','addr':'서울 서초구','depts':['외과','정형외과','비뇨의학과'],'type':'병원','rating':4.4},
    ],
    '충청권': [
        {'name':'대전중앙내과','addr':'대전 서구','depts':['내과','신경과','정신건강의학과'],'type':'의원','rating':4.3},
        {'name':'충청종합병원','addr':'충남 천안시','depts':['외과','내과','이비인후과'],'type':'병원','rating':4.1},
        {'name':'세종으뜸의원','addr':'세종시','depts':['내과','피부과','안과'],'type':'의원','rating':4.2},
    ],
    '호남권': [
        {'name':'광주제일의원','addr':'광주 북구','depts':['내과','외과','이비인후과'],'type':'의원','rating':4.2},
        {'name':'전남중앙병원','addr':'전남 목포시','depts':['신경과','내과','정형외과'],'type':'병원','rating':4.0},
        {'name':'전북현대의원','addr':'전북 전주시','depts':['내과','피부과','비뇨의학과'],'type':'의원','rating':4.3},
    ],
    '대구경북권': [
        {'name':'대구중앙의원','addr':'대구 중구','depts':['내과','외과','이비인후과'],'type':'의원','rating':4.4},
        {'name':'경북종합병원','addr':'경북 포항시','depts':['정형외과','신경과','내과'],'type':'병원','rating':4.1},
    ],
    '부산경남권': [
        {'name':'부산해운대의원','addr':'부산 해운대구','depts':['내과','피부과','이비인후과'],'type':'의원','rating':4.3},
        {'name':'경남중앙병원','addr':'경남 창원시','depts':['외과','신경과','비뇨의학과'],'type':'병원','rating':4.2},
        {'name':'울산현대의원','addr':'울산 남구','depts':['내과','정형외과','안과'],'type':'의원','rating':4.1},
    ],
    '강원권': [
        {'name':'강원중앙의원','addr':'강원 춘천시','depts':['내과','외과','이비인후과'],'type':'의원','rating':4.2},
        {'name':'원주세브란스병원분원','addr':'강원 원주시','depts':['신경과','내과','외과'],'type':'병원','rating':4.4},
    ],
    '제주권': [
        {'name':'제주중앙의원','addr':'제주시','depts':['내과','피부과','외과'],'type':'의원','rating':4.3},
        {'name':'서귀포의료원','addr':'서귀포시','depts':['내과','외과','신경과'],'type':'병원','rating':4.1},
    ],
}

# Track B: 3차 상급종합병원 (권역별, 진료과별 전문의 포함)
TRACK_B_DB = {
    '수도권': [
        {'name':'서울대학교병원','addr':'서울 종로구','specialty':['신경외과','심장외과','종양내과','혈액종양과'],'level':'상급종합','prof_count':320},
        {'name':'세브란스병원','addr':'서울 서대문구','specialty':['신경과','소화기내과','심장내과','내분비내과'],'level':'상급종합','prof_count':280},
        {'name':'삼성서울병원','addr':'서울 강남구','specialty':['암센터','뇌신경센터','심혈관센터','희귀질환센터'],'level':'상급종합','prof_count':300},
        {'name':'서울아산병원','addr':'서울 송파구','specialty':['간이식','심장이식','신경외과','소아청소년과'],'level':'상급종합','prof_count':310},
    ],
    '충청권': [
        {'name':'충남대학교병원','addr':'대전 중구','specialty':['신경외과','혈액종양과','심장내과'],'level':'상급종합','prof_count':120},
        {'name':'건양대학교병원','addr':'대전 서구','specialty':['안과','이비인후과','내과'],'level':'상급종합','prof_count':95},
    ],
    '호남권': [
        {'name':'전남대학교병원','addr':'광주 동구','specialty':['혈액종양과','신경과','심장내과'],'level':'상급종합','prof_count':140},
        {'name':'조선대학교병원','addr':'광주 동구','specialty':['외과','이비인후과','비뇨의학과'],'level':'상급종합','prof_count':110},
    ],
    '대구경북권': [
        {'name':'경북대학교병원','addr':'대구 중구','specialty':['신경외과','혈액종양과','소화기내과'],'level':'상급종합','prof_count':150},
        {'name':'계명대학교동산병원','addr':'대구 달서구','specialty':['심장내과','내분비내과','신경과'],'level':'상급종합','prof_count':130},
    ],
    '부산경남권': [
        {'name':'부산대학교병원','addr':'부산 서구','specialty':['신경외과','혈액종양과','심장외과'],'level':'상급종합','prof_count':160},
        {'name':'양산부산대학교병원','addr':'경남 양산시','specialty':['소화기내과','내분비내과','신경과'],'level':'상급종합','prof_count':140},
    ],
    '강원권': [
        {'name':'강원대학교병원','addr':'강원 춘천시','specialty':['신경과','외과','내과'],'level':'상급종합','prof_count':80},
        {'name':'한림대학교춘천성심병원','addr':'강원 춘천시','specialty':['심장내과','신경외과','비뇨의학과'],'level':'상급종합','prof_count':75},
    ],
    '제주권': [
        {'name':'제주대학교병원','addr':'제주시','specialty':['신경과','내과','외과','혈액종양과'],'level':'상급종합','prof_count':60},
    ],
}

def get_region(sido: str) -> str:
    for region, sidos in REGIONS.items():
        for s in sidos:
            if s in sido:
                return region
    return '수도권'

def query_track_a(region: str, dept_keywords: list) -> list:
    hospitals = TRACK_A_DB.get(region, TRACK_A_DB['수도권'])
    scored = []
    for h in hospitals:
        score = sum(1 for kw in dept_keywords if any(kw in d for d in h['depts']))
        scored.append((score, h))
    scored.sort(key=lambda x: (-x[0], -x[1]['rating']))
    return [h for _, h in scored[:3]]

def query_track_b(region: str, disease_keywords: list) -> list:
    hospitals = TRACK_B_DB.get(region, TRACK_B_DB['수도권'])
    scored = []
    for h in hospitals:
        score = sum(1 for kw in disease_keywords if any(kw in s for s in h['specialty']))
        scored.append((score, h))
    scored.sort(key=lambda x: (-x[0], -x[1]['prof_count']))
    return [h for _, h in scored[:3]]
