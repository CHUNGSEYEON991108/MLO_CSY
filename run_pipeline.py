# run_pipeline.py
import argparse
import pandas as pd
import os
from mlops_csy.loader import load_data
from mlops_csy.explorer import explore_data
from mlops_csy.visualizer import count_unique_values, plot_distribution
from mlops_csy.model_utils import train_model, predict_and_save

if __name__ == '__main__':
    # 스크립트가 있는 디렉토리의 절대 경로를 가져옴
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # CLI에서 실행 시 입력받을 인자들
    parser = argparse.ArgumentParser(description='신용카드 채무 불이행 예측 파이프라인')
    parser.add_argument('--train_path', type=str, default=os.path.join(base_dir, 'data', 'train.csv'), help='학습 데이터 경로')
    parser.add_argument('--test_path', type=str, default=os.path.join(base_dir, 'data', 'test.csv'), help='테스트 데이터 경로')
    parser.add_argument('--output_path', type=str, default=os.path.join(base_dir, 'output', 'submission.csv'), help='예측 결과 저장 경로')
    args = parser.parse_args()

    # 데이터 로드
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)

    # 데이터 탐색 및 시각화
    explore_data(train_df)
    count_unique_values(train_df, '대출 목적')
    plot_distribution(train_df, '연간 소득')

    # 모델 학습 및 예측
    model = train_model(train_df)
    predict_and_save(model, test_df, args.output_path)
