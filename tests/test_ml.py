"""생존 예측 모델 테스트."""


def test_cv_accuracy_above_baseline(model):
    # 다수 클래스(사망 61.6%) 예측보다 유의미하게 높아야 한다
    assert model.cv_accuracy is not None
    assert model.cv_accuracy > 0.75


def test_predict_returns_valid_probability(model):
    result = model.predict(pclass=1, sex="female", age=30, fare=80.0)
    assert 0.0 <= result.probability <= 1.0
    assert isinstance(result.survived, bool)


def test_prediction_direction_matches_domain_knowledge(model):
    # 1등석 여성은 3등석 남성보다 생존 확률이 높아야 한다 (여성/상위등급 우선 구조)
    female_first = model.predict(pclass=1, sex="female", age=30, fare=80.0)
    male_third = model.predict(pclass=3, sex="male", age=30, fare=8.0)
    assert female_first.probability > male_third.probability


def test_predict_handles_missing_fare(model):
    # fare가 None이어도 파이프라인 imputer가 처리해야 한다
    result = model.predict(pclass=2, sex="male", age=40, fare=None)
    assert 0.0 <= result.probability <= 1.0
