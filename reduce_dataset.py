import os
import random
import shutil

SOURCE = "processed_dataset"
DEST = "small_dataset"
LIMIT = 100  # max files per language per class

for category in ["human", "ai"]:
    category_path = os.path.join(SOURCE, category)

    for lang in os.listdir(category_path):
        src_lang = os.path.join(category_path, lang)
        dest_lang = os.path.join(DEST, category, lang)

        os.makedirs(dest_lang, exist_ok=True)

        files = [f for f in os.listdir(src_lang) if f.endswith(".wav")]

        if len(files) == 0:
            continue

        selected = random.sample(files, min(LIMIT, len(files)))

        for f in selected:
            shutil.copy(
                os.path.join(src_lang, f),
                os.path.join(dest_lang, f)
            )

        print(f"{category}/{lang}: {len(selected)} files")

print("✅ Reduced dataset created in small_dataset/")
