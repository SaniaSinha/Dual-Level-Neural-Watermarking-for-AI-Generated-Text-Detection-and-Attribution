from dual_system import DualLevelWatermarkingSystem


def print_block(title, data):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    for key, value in data.items():
        print(f"{key}: {value}")


def main():
    system = DualLevelWatermarkingSystem()

    text = (
        "Artificial intelligence significantly improves modern healthcare systems "
        "by supporting faster diagnosis, better decision making, and efficient patient care."
    )

    text_id = 2026
    source_id = "0101"

    detection = system.detect_sentence(text)
    print_block("AI DETECTION", detection)

    encoding = system.encode(
        text=text,
        text_id=text_id,
        source_id=source_id
    )

    print_block("ENCODING RESULT", encoding)

    if not encoding["success"]:
        return

    decoding = system.decode(
        watermarked_text=encoding["watermarked_text"],
        text_id=text_id
    )

    print_block("DECODING RESULT", decoding)

    print("\nFINAL SUMMARY")
    print("=" * 70)
    print("Original Source:", encoding["source_name"])
    print("Recovered Source:", decoding["source_name"])
    print("Original Watermark:", encoding["full_watermark"])
    print("Recovered Watermark:", decoding["reconstructed_watermark"])
    print("Ownership Verified:", decoding["ownership_verified"])


if __name__ == "__main__":
    main()
