import pytest

from acb.tools import amortize, calculate_loan_from_message, csv_fast_path


def test_amortize_standard_loan():
    result = amortize(10000, 8.5, 60)
    assert result["principal"] == 10000
    assert result["term_months"] == 60
    assert result["monthly_payment"] > 0
    assert result["total_interest"] > 0


def test_amortize_zero_rate_splits_principal_evenly():
    result = amortize(1200, 0, 12)
    assert result["monthly_payment"] == 100.0
    assert result["total_interest"] == 0.0


def test_amortize_rejects_non_positive_principal():
    with pytest.raises(ValueError):
        amortize(0, 5, 12)
    with pytest.raises(ValueError):
        amortize(-100, 5, 12)


def test_amortize_rejects_negative_rate():
    with pytest.raises(ValueError):
        amortize(1000, -1, 12)


def test_amortize_rejects_non_positive_term():
    with pytest.raises(ValueError):
        amortize(1000, 5, 0)


def test_csv_fast_path_is_case_and_whitespace_insensitive():
    # qa_data.csv is a large real dataset; just confirm the lookup path
    # normalizes input rather than asserting on specific content.
    from acb.tools import CSV_REPLIES

    if not CSV_REPLIES:
        pytest.skip("qa_data.csv not present in this environment")
    sample_question = next(iter(CSV_REPLIES))
    assert csv_fast_path(f"  {sample_question.upper()}  ") == CSV_REPLIES[sample_question]


def test_csv_fast_path_returns_none_for_unknown_question():
    assert csv_fast_path("some question that will never be in the csv xyz123") is None


def test_calculate_loan_from_message_extracts_amount_rate_and_term():
    result = calculate_loan_from_message("what would a $10,000 loan at 8.5% for 5 years cost?", default_term_years=5)
    assert result is not None
    assert result["principal"] == 10000.0
    assert result["annual_rate"] == 8.5
    assert result["term_months"] == 60
    assert result["is_estimate"] is True


def test_calculate_loan_from_message_handles_k_suffix_and_default_term():
    result = calculate_loan_from_message("a 300k mortgage at 6.5%", default_term_years=30)
    assert result is not None
    assert result["principal"] == 300000.0
    assert result["term_months"] == 360


def test_calculate_loan_from_message_returns_none_without_a_rate():
    assert calculate_loan_from_message("how much would a $10,000 loan cost?", default_term_years=5) is None


def test_calculate_loan_from_message_returns_none_without_an_amount():
    assert calculate_loan_from_message("what's a good interest rate, around 8%?", default_term_years=5) is None
