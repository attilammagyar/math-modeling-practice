import re


CATEGORIES_PATTERN = re.compile(
    r"\s*;*\s*([0-9]+)\s+[–-]\s+(.*?)(?=(\s*;*\s*[0-9]+\s+[–-]\s+)|$)(.*)$"
)


def main():
    with open("data/features.tsv") as f:
        print_features(parse_features(f))


def print_features(features):
    print("{")

    for name, (descr, categories) in features.items():
        print(f"    {name!r}: [{descr!r}, ", end="")

        if not categories:
            print("{}", end="")
        else:
            print("{")

            categories = sorted(categories.items(), key=lambda e: e[0])

            for cat_id, cat_descr in categories:
                spaces = " " * max(0, 6 - len(str(cat_id)))
                print(f"        {cat_id}: {spaces}{cat_descr!r},")

            print("    }", end="")

        print("],")
        print("")

    print("}")


def parse_features(lines):
    features = {}

    for line in lines:
        line = line.strip()

        if not line:
            continue

        name, descr, categories = parse_feature(line)
        features[name] = (descr, categories)

    return features


def parse_feature(line):
    name, role, type_, dem, descr, units, na = line.split("\t")
    categories = {}
    match_ = CATEGORIES_PATTERN.match(descr)

    if type_ == "Integer" and match_:
        while match_:
            key = int(match_[1])
            categories[key] = f"({key}) {match_[2]}"
            descr = match_[4] if match_ else ""
            match_ = CATEGORIES_PATTERN.match(descr)

        descr = ""

    elif descr.lower() == name.lower():
        descr = ""

    if dem and dem.lower() != name.lower():
        descr = f"({dem}) {descr}" if descr else f"({dem})"

    return name, descr, categories


if __name__ == "__main__":
    main()
