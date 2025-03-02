from pathlib import Path

import random
import re
import shutil


def replace_random_digit(original: str, modified: str):
    with open(original, "r", encoding="utf-8") as file:
        header = file.readline()
        header += file.readline()
        print(header)
        content = file.read()
        print(content)

    positions = [match.start() for match in re.finditer(r"\d", content)]
    assert positions, "file should contain digits"

    index = random.choice(positions)
    old = content[index]
    new = random.choice([digit for digit in "0123456789" if digit != old])

    content = content[:index] + new + content[index + 1 :]

    print("NEW")

    # with open(modified, "w", encoding="utf-8") as file:
    #     print(header)
    #     print(content)
    #     file.write(header)
    #     file.write(content)
    shutil.copyfile(original, modified)
