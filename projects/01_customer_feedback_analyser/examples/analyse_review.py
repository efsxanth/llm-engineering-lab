from customer_feedback_analyser.analyser import analyse_feedback


def main() -> None:
    """Demonstrate customer feedback analysis."""

    customer_review = (
        "I bought these headphones for my brother's birthday. "
        "They arrived in three days. "
        "The sound quality is excellent, although they are "
        "a little expensive."
    )

    result = analyse_feedback(customer_review)

    print("Customer review:")
    print(customer_review)

    print("\nAnalysis:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
