# run_pipeline.py
import argparse
import pandas as pd
import os
from mlops_csy.loader import load_data
from mlops_csy.explorer import explore_data
from mlops_csy.visualizer import count_unique_values, plot_distribution
from mlops_csy.model_utils import train_model, predict_and_save

def main():
    """메인 실행 함수"""
    # 스크립트가 있는 디렉토리의 절대 경로를 가져옴
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # CLI에서 실행 시 입력받을 인자들
    parser = argparse.ArgumentParser(description='신용카드 채무 불이행 예측 파이프라인')
    parser.add_argument('--train_path', type=str, 
                       default=os.path.join(base_dir, 'data', 'train.csv'),
                       help='학습 데이터 경로')
    parser.add_argument('--test_path', type=str,
                       default=os.path.join(base_dir, 'data', 'test.csv'),
                       help='테스트 데이터 경로')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(base_dir, 'output'),
                       help='결과물 저장 디렉토리')
    parser.add_argument('--skip_eda', action='store_true',
                       help='EDA 단계 스킵 여부')
    
    args = parser.parse_args()
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n📊 채무 불이행 예측 파이프라인 시작")
    print(f"학습 데이터: {args.train_path}")
    print(f"테스트 데이터: {args.test_path}")
    print(f"결과물 저장 위치: {args.output_dir}")
    
    # 데이터 로드
    print("\n📂 데이터 로드 중...")
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    
    # EDA 수행 (선택적)
    if not args.skip_eda:
        print("\n📈 데이터 탐색 분석 수행 중...")
        explore_data(train_df)
        count_unique_values(train_df, '대출 목적')
        plot_distribution(train_df, '연간 소득')
    
    # 모델 학습
    model = train_model(train_df)
    
    # 예측 및 저장
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    predict_and_save(model, test_df, submission_path)
    
    print("\n✨ 파이프라인 실행 완료!")

if __name__ == '__main__':
    main()
