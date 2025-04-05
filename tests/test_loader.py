import pytest
import pandas as pd
import os
from mlops_csy.loader import load_data

def test_load_data_success():
    """데이터 로드 성공 테스트"""
    # 테스트용 임시 데이터 생성
    test_data = pd.DataFrame({
        'UID': [1, 2, 3],
        '채무 불이행 여부': [0, 1, 0]
    })
    
    # 임시 파일로 저장
    test_path = 'test_data.csv'
    test_data.to_csv(test_path, index=False)
    
    try:
        # 데이터 로드 테스트
        loaded_data = load_data(test_path)
        assert isinstance(loaded_data, pd.DataFrame)
        assert len(loaded_data) == 3
        assert 'UID' in loaded_data.columns
        assert '채무 불이행 여부' in loaded_data.columns
    finally:
        # 테스트 후 임시 파일 삭제
        if os.path.exists(test_path):
            os.remove(test_path)

def test_load_data_file_not_found():
    """존재하지 않는 파일 로드 시 에러 테스트"""
    with pytest.raises(FileNotFoundError):
        load_data('non_existent_file.csv') 