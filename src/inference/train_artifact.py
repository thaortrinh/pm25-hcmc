from __future__ import annotations

from src.inference.artifact import train_deployable_artifact


def main() -> None:
    model_path, metadata_path = train_deployable_artifact(force=True)
    print(f"Saved model artifact: {model_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
