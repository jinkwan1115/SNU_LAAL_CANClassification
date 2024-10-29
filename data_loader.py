import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DrivingDataLoader_Supervised:
    def __init__(self, file_path, window_size=10, step_size=1):
        self.file_path = file_path
        self.window_size = window_size
        self.step_size = step_size
        self.df = None
        self.class_windows = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        # 데이터를 불러오고 Record_ID 열을 생성
        self.df = pd.read_csv(self.file_path)
        self.df['Record_ID'] = (self.df['Time(s)'] == 1).cumsum()
        self.df['Class'] = self.label_encoder.fit_transform(self.df['Class'])

    def sliding_window_by_record(self):
        # Record_ID 별로 슬라이딩 윈도우 생성
        record_windows = {}  # Record_ID별 윈도우 리스트를 저장할 딕셔너리

        for record_id, group in self.df.groupby('Record_ID'):
            windows = []  # 각 Record_ID별 슬라이딩 윈도우를 저장할 리스트
            max_index = len(group) - self.window_size + 1  # 가능한 최대 인덱스
            
            for start in range(0, max_index, self.step_size):
                window = group.iloc[start:start + self.window_size].reset_index(drop=True)
                windows.append(window)
            
            # Record_ID별 윈도우 리스트를 딕셔너리에 저장
            record_windows[record_id] = windows

        return record_windows

    def create_class_windows(self):
        # Class 별 윈도우 생성
        record_windows = self.sliding_window_by_record()
        class_windows = {}

        for record_id, windows in record_windows.items():
            for window in windows:
                y_label = window['Class'].iloc[0]  # Class 열의 첫 번째 값을 y로 설정
                X_window = window.drop(columns=['Record_ID', 'PathOrder', 'Class', 'Time(s)'])  # X 값으로 사용할 열만 남김
                
                # y_label을 키로 하고, X_window 리스트에 추가
                if y_label not in class_windows:
                    class_windows[y_label] = []  # y_label이 처음 등장하면 리스트 초기화
                class_windows[y_label].append(X_window)  # 모든 윈도우 추가

        self.class_windows = class_windows
        return class_windows

    def get_class_windows(self):
        # 클래스 윈도우 반환
        if self.class_windows is None:
            self.create_class_windows()
        return self.class_windows
