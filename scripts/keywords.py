# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pypinyin",
# ]
# [tool.uv]
# exclude-newer = "2025-05-06T00:00:00Z"
# ///

import argparse
from pathlib import Path
from typing import List, Union
from pypinyin import pinyin
from pypinyin.contrib.tone_convert import to_initials, to_finals_tone


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text",
        type=str,
        required=False,
        default="my-keywords.txt",
        help="""Path to the input texts.

        Each line in the texts contains the original phrase, it might also contain some
        extra items, for example, the boosting score (startting with :), the triggering
        threshold (startting with #, only used in keyword spotting task) and the original
        phrase (startting with @). Note: extra items will be kept in the output.

        example input 1 (tokens_type = ppinyin):

        小爱同学 :2.0 #0.6 @小爱同学
        你好问问 :3.5 @你好问问
        小艺小艺 #0.6 @小艺小艺

        example output 1:

        x iǎo ài t óng x ué :2.0 #0.6 @小爱同学
        n ǐ h ǎo w èn w èn :3.5 @你好问问
        x iǎo y ì x iǎo y ì #0.6 @小艺小艺

        example input 2 (tokens_type = bpe):

        HELLO WORLD :1.5 #0.4
        HI GOOGLE :2.0 #0.8
        HEY SIRI #0.35

        example output 2:

        ▁HE LL O ▁WORLD :1.5 #0.4
        ▁HI ▁GO O G LE :2.0 #0.8
        ▁HE Y ▁S I RI #0.35
        """,
    )

    parser.add_argument(
        "--tokens",
        type=str,
        required=False,
        default="tokens.txt",
        help="The path to tokens.txt.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default="keywords.txt",
        help="Path where the encoded tokens will be written to.",
    )

    return parser.parse_args()


def text2token(
    texts: List[str],
    tokens: str,
    output_ids: bool = False,
):
    """将文本转换为 token 列表"""
    assert Path(tokens).is_file(), f"File not exists, {tokens}"
    tokens_table = {}
    with open(tokens, "r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            assert len(toks) == 2, len(toks)
            assert toks[0] not in tokens_table, f"Duplicate token: {toks} "
            tokens_table[toks[0]] = int(toks[1])

    texts_list: List[List[str]] = []
    for txt in texts:
        res = []
        py = [x[0] for x in pinyin(txt)]
        for x in py:
            initial = to_initials(x, strict=False)
            final = to_finals_tone(x, strict=False)
            if initial == "" and final == "":
                res.append(x)
            else:
                if initial != "":
                    res.append(initial)
                if final != "":
                    res.append(final)
        texts_list.append(res)

    result: List[List[Union[int, str]]] = []
    for text in texts_list:
        text_list = []
        contain_oov = False
        for txt in text:
            if txt in tokens_table:
                text_list.append(tokens_table[txt] if output_ids else txt)
            else:
                print(
                    f"Can't find token {txt} in token table, check your "
                    f"tokens.txt see if {txt} in it. skipping text : {text}."
                )
                contain_oov = True
                break
        if contain_oov:
            continue
        else:
            result.append(text_list)
    return result


def main() -> None:
    args = get_args()

    texts = []
    # extra information like boosting score (start with :), triggering threshold (start with #)
    # original keyword (start with @)
    extra_info = []
    with open(args.text, "r", encoding="utf8") as f:
        for line in f:
            extra = []
            text = []
            toks = line.strip().split()
            if len(toks) == 1:
                text.append(toks[0])
                extra.append('@'+toks[0])
            else:
                for tok in toks:
                    if tok[0] == ":" or tok[0] == "#" or tok[0] == "@":
                        extra.append(tok)
                    else:
                        text.append(tok)
            texts.append(" ".join(text))
            extra_info.append(extra)
    encoded_texts = text2token(
        texts,
        tokens=args.tokens,
    )
    with open(args.output, "w", encoding="utf8") as f:
        for i, txt in enumerate(encoded_texts):
            txt += extra_info[i]
            f.write(" ".join(txt) + "\n")
    print(f"✅ 唤醒词已保存到 {args.output}")


if __name__ == "__main__":
    main()
