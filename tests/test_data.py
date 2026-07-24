"""데이터 로딩·피처 엔지니어링 테스트."""

from titanic_agent.data import dataset_overview


def test_load_dataset_shape(df):
    assert len(df) == 891
    assert "Survived" in df.columns


def test_derived_features(df):
    assert "Familysize" in df.columns
    assert "AgeGroup" in df.columns
    # Familysize = SibSp + Parch
    sample = df.iloc[0]
    assert sample["Familysize"] == sample["SibSp"] + sample["Parch"]


def test_dataset_overview(df):
    overview = dataset_overview(df)
    assert overview["num_rows"] == 891
    # 전체 생존율은 알려진 값(약 38.4%)과 일치해야 한다
    assert abs(overview["survival_rate"] - 0.3838) < 0.001
    # Age 결측치 177건이 보고되어야 한다
    assert overview["missing_values"]["Age"] == 177
