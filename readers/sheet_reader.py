from pathlib import Path

from llama_index.core import Document


def load_sheets(sheet_files: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for path in sheet_files:
        try:
            ext = path.suffix.lower()
            if ext == ".csv":
                docs.extend(_load_csv(path))
            elif ext in {".xlsx", ".xls"}:
                docs.extend(_load_excel(path))
            print(f"[sheet_reader] {path.name} — OK ({len(docs)} docs so far)")
        except Exception as e:
            print(f"[sheet_reader] Failed to load {path.name}: {e}")
    return docs


def _load_csv(path: Path) -> list[Document]:
    import pandas as pd

    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    return _df_to_docs(df, path.name)


def _load_excel(path: Path) -> list[Document]:
    import pandas as pd

    df = pd.read_excel(path, header=0)
    return _df_to_docs(df, path.name)


def _df_to_docs(df, filename: str) -> list[Document]:
    docs: list[Document] = []
    for i, row in df.iterrows():
        text = " | ".join(
            f"{col}: {val}"
            for col, val in row.items()
            if str(val).strip() not in {"", "nan", "NaN", "None"}
        )
        if text.strip():
            docs.append(
                Document(
                    text=text,
                    metadata={"file_name": filename, "row": int(i)},
                )
            )
    return docs
