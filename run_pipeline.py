# run_pipeline.py
import argparse
import pandas as pd
import os
from mlops_csy.loader import load_data
from mlops_csy.explorer import explore_data
from mlops_csy.visualizer import count_unique_values, plot_distribution
from mlops_csy.model_utils import train_model, predict_and_save
import matplotlib.pyplot as plt

def setup_directories(output_dir):
    """필요한 디렉토리들을 생성합니다."""
    os.makedirs(output_dir, exist_ok=True)
    
def run_pipeline(train_path, test_path, output_dir, skip_viz=False):
    """전체 파이프라인을 실행합니다.
    
    Args:
        train_path (str): 학습 데이터 경로
        test_path (str): 테스트 데이터 경로
        output_dir (str): 결과물 저장 디렉토리
        skip_viz (bool): 시각화 스킵 여부
    """
    # 디렉토리 설정
    setup_directories(output_dir)
    
    # 데이터 로드
    print("데이터를 로드하고 있습니다...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 데이터 탐색 및 시각화
    if not skip_viz:
        print("데이터 탐색 및 시각화를 수행합니다...")
        explore_data(train_df)
        count_unique_values(train_df, '대출 목적')
        plot_distribution(train_df, '연간 소득')
        
        # 시각화 결과 저장
        plt.savefig(os.path.join(output_dir, 'distributions.png'))
    
    # 모델 학습 및 예측
    print("모델 학습을 시작합니다...")
    model = train_model(train_df)
    
    print("예측을 수행하고 결과를 저장합니다...")
    submission_path = os.path.join(output_dir, 'submission.csv')
    predict_and_save(model, test_df, submission_path)
    
    print(f"\n작업이 완료되었습니다!")
    print(f"결과물 저장 위치: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='신용카드 채무 불이행 예측 파이프라인')
    
    # 필수 인자
    parser.add_argument('train_path', type=str, help='학습 데이터 경로 (train.csv)')
    parser.add_argument('test_path', type=str, help='테스트 데이터 경로 (test.csv)')
    
    # 선택적 인자
    parser.add_argument('--output_dir', type=str, default='output',
                      help='결과물 저장 디렉토리 (기본값: output)')
    parser.add_argument('--skip_viz', action='store_true',
                      help='시각화 단계 스킵 여부')
    
    args = parser.parse_args()
    
    # 파이프라인 실행
    run_pipeline(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        skip_viz=args.skip_viz
    )
