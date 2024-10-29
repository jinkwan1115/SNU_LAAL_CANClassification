import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random

class DrivingDataLoader_Supervised:
    def __init__(self, file_path, window_size=10, step_size=1, test_size=0.2, val_size=0.1):
        self.file_path = file_path
        self.window_size = window_size
        self.step_size = step_size
        self.test_size = test_size
        self.val_size = val_size
        self.df = None
        self.class_windows = None
        self.label_encoder = LabelEncoder()
        self.train_windows = None
        self.val_windows = None
        self.test_windows = None

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

    def split_data(self):
        # Class 별로 train, validation, test set 생성
        if self.class_windows is None:
            self.create_class_windows()
        
        train_set, val_set, test_set = [], [], []

        for y_label, windows in self.class_windows.items():
            # 각 클래스의 윈도우 데이터를 섞은 후 train, val, test로 분할
            windows_train, windows_test = train_test_split(windows, test_size=self.test_size, random_state=42)
            train_split, val_split = train_test_split(windows_train, test_size=self.val_size / (1 - self.test_size), random_state=42)

            train_set.extend([(X, y_label) for X in train_split])
            val_set.extend([(X, y_label) for X in val_split])
            test_set.extend([(X, y_label) for X in windows_test])

        self.train_windows = train_set
        self.val_windows = val_set
        self.test_windows = test_set

        return train_set, val_set, test_set

    def get_splits(self):
        # Train, Validation, Test set 반환
        if self.train_windows is None or self.val_windows is None or self.test_windows is None:
            self.split_data()
        return self.train_windows, self.val_windows, self.test_windows


class DrivingDataLoader_SemiSupervised:
    def __init__(self, file_path, window_size=10, step_size=1, unlabeled_ratio=0.2,
                 train_ratio=0.7, validation_ratio=0.15, test_ratio=0.15):
        self.file_path = file_path
        self.window_size = window_size
        self.step_size = step_size
        self.unlabeled_ratio = unlabeled_ratio
        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.df = None
        self.class_windows = None
        self.unlabeled_windows = None
        self.train_labeled = None
        self.train_unlabeled = None
        self.validation_set = None
        self.test_set = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        # 데이터를 불러오고 Record_ID 열을 생성
        self.df = pd.read_csv(self.file_path)
        self.df['Record_ID'] = (self.df['Time(s)'] == 1).cumsum()
        self.df['Class'] = self.label_encoder.fit_transform(self.df['Class'])

    def sliding_window_by_record(self):
        # Record_ID 별로 슬라이딩 윈도우 생성
        record_windows = {}

        for record_id, group in self.df.groupby('Record_ID'):
            windows = []
            max_index = len(group) - self.window_size + 1
            
            for start in range(0, max_index, self.step_size):
                window = group.iloc[start:start + self.window_size].reset_index(drop=True)
                windows.append(window)
            
            record_windows[record_id] = windows

        return record_windows

    def create_class_windows(self):
        # Class 별 윈도우 생성
        record_windows = self.sliding_window_by_record()
        class_windows = {}

        for record_id, windows in record_windows.items():
            for window in windows:
                y_label = window['Class'].iloc[0]
                X_window = window.drop(columns=['Record_ID', 'PathOrder', 'Class', 'Time(s)'])
                
                if y_label not in class_windows:
                    class_windows[y_label] = []
                class_windows[y_label].append((X_window, y_label))

        self.class_windows = class_windows
        return class_windows

    def split_data(self):
        # Train, Validation, Test 세트를 생성
        if self.class_windows is None:
            self.create_class_windows()

        train_labeled = []
        train_unlabeled = []
        validation_set = []
        test_set = []

        for y_label, windows in self.class_windows.items():
            total_windows = len(windows)
            train_count = int(total_windows * self.train_ratio)
            validation_count = int(total_windows * self.validation_ratio)
            
            # 무작위 셔플 후 분할
            random.shuffle(windows)
            train_windows = windows[:train_count]
            validation_set.extend(windows[train_count:train_count + validation_count])
            test_set.extend(windows[train_count + validation_count:])

            # 지정된 비율로 레이블 제거하여 unlabeled 데이터 생성
            num_unlabeled = int(len(train_windows) * self.unlabeled_ratio)
            train_unlabeled.extend([(X_window, None) for X_window, _ in train_windows[:num_unlabeled]])
            train_labeled.extend(train_windows[num_unlabeled:])

        self.train_labeled = train_labeled
        self.train_unlabeled = train_unlabeled
        self.validation_set = validation_set
        self.test_set = test_set
        return train_labeled, train_unlabeled, validation_set, test_set

    def get_splits(self):
        # Train, Validation, Test 세트 반환
        if self.train_labeled is None or self.train_unlabeled is None or self.validation_set is None or self.test_set is None:
            self.split_data()
        return self.train_labeled, self.train_unlabeled, self.validation_set, self.test_set