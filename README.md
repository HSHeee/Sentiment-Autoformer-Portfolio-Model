# SentimentAutoformerPortfolio

## TODO
단계	주요 작업
1	목표 정의, 적용시장 선정
2	Bloomberg/TREN 데이터 수집
3	정규화, LASSO, 감정 통합
4	NSAutoformer 모델 구현
5	평가 지표 기반 비교 실험
6	결과 해석 및 시각화



### ✅ 1단계 완료: 프로젝트 구성 요약
항목	내용
예측 대상	업종별 ETF (예: XLK, XLE, XLF)
감정지수 이슈	개별 기업 감정지수만 존재 → ETF 감정지수는 구성 종목 기반 가중평균 등으로 생성 예정
예측 목표	종가, 수익률, 방향성 모두 테스트 → 포트폴리오 성과 극대화 기준으로 선택 예정
예측 단위	T+1, T+5 등 다양한 시차 실험
사용 데이터	Bloomberg 가격/거래량, TA-Lib 기반 기술지표, TREN 뉴스 감정지수, SNS 감정지수



