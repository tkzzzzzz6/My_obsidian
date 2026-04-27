#!/usr/bin/env python3
"""
仿射密码破解脚本
包含：频率分析、数学推导、暴力破解三种方法
加密公式: E(x) = (a*x + b) mod 26
"""

from collections import Counter
import re

# ========== 1. 密文输入 ==========
CIPHERTEXT = """Xzze qjpuy jwi ynez qnm uyz qlulmz. Fjxz bnlm fndu dpwrzmz imzjfd, jwi tyzw uyz neenmulwpupzd rnfz, uyzb tpoo qphyu qnm uyzf. Pu fjb ujxz j dzjdnw nm fnmz, alu uyz zwipwh tpoo wnu ryjwhz. Jfapupnw, azdu, azrnfz j mzjopub. Jw lwrzmujpw qlulmz, nwob nwz duze ju j upfz, uyz ynez rjw mzjopsz uyz imzjf nq uyz yphyzdu. Tz fldu umzjdlmz uyz imzjf, un emnuzru pu j dzjdnw, ozu pu pw uyz yzjmu vlpzuob hzmfpwjo. Yntzczm, tz yjcz un hzwuob emnuzru nlm yzjmud izze zkezrujupnwd, dontob imzjf, tpoo jrypzcz wzt opqz."""

# ========== 2. 频率分析 ==========

def frequency_analysis(text):
    """统计字母频率并排序"""
    letters = ''.join(c for c in text if c.isalpha())
    freq = Counter(letters.lower())
    total = len(letters)
    print("=" * 50)
    print("密文字母频率统计（Top 10）")
    print("-" * 50)
    for letter, count in freq.most_common(10):
        print(f"  {letter}: {count:3d}  ({count/total*100:5.1f}%)")
    print(f"  总字母数: {total}")
    return freq.most_common(10)


# ========== 3. 数学推导求密钥 ==========

def mod_inverse(a, m):
    """求 a 在模 m 下的乘法逆元"""
    for i in range(1, m):
        if (a * i) % m == 1:
            return i
    return None

def derive_key_from_frequency(top_freq):
    """
    假设英文最高频字母 e(4) 对应密文最高频字母 z(25)
    假设英文次高频字母 t(19) 对应密文次高频字母 u(20)
    解方程组求 a, b
    """
    # z = 25, u = 20, e = 4, t = 19
    z, u = 25, 20
    e, t = 4, 19

    print("\n" + "=" * 50)
    print("频率分析推导密钥")
    print("-" * 50)
    print(f"假设: e({e}) -> z({z}), t({t}) -> u({u})")
    print(f"\n方程组:")
    print(f"  {e}a + b ≡ {z}  (mod 26)")
    print(f"  {t}a + b ≡ {u}  (mod 26)")

    # 相减: (e-t)a ≡ (z-u) (mod 26)
    coeff_a = (e - t) % 26
    rhs = (z - u) % 26
    print(f"\n相减: {coeff_a}a ≡ {rhs} (mod 26)")

    # 求逆元
    inv = mod_inverse(coeff_a, 26)
    if inv is None:
        print("逆元不存在，假设有误")
        return None

    a = (rhs * inv) % 26
    b = (z - e * a) % 26
    print(f"\n{coeff_a}的逆元: {inv}")
    print(f"解得: a = {a}, b = {b}")

    # 验证
    check1 = (e * a + b) % 26
    check2 = (t * a + b) % 26
    print(f"\n验证:")
    print(f"  e({e}) -> {check1} (应为{z}) {'✓' if check1 == z else '✗'}")
    print(f"  t({t}) -> {check2} (应为{u}) {'✓' if check2 == u else '✗'}")

    return (a, b)


# ========== 4. 暴力破解 ==========

VALID_A = [1, 3, 5, 7, 9, 11, 15, 17, 19, 21, 23, 25]  # 与26互质的数

COMMON_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it',
    'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but',
    'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will',
    'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out',
    'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can',
    'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people', 'into',
    'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than',
    'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
    'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well',
    'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day',
    'most', 'us', 'is', 'was', 'are', 'were', 'been', 'has', 'had', 'did',
    'does', 'keep', 'hope', 'future', 'make', 'dreams', 'opportunities',
    'come', 'they', 'may', 'take', 'season', 'more', 'but', 'ending',
    'change', 'ambition', 'best', 'become', 'reality', 'uncertain', 'only',
    'step', 'realize', 'dream', 'highest', 'must', 'treasure', 'protect',
    'heart', 'quietly', 'however', 'gently', 'protect', 'deep', 'slowly',
    'achieve', 'life', 'faith'
}

def affine_decrypt(ciphertext, a, b):
    """仿射解密: D(y) = a_inv * (y - b) mod 26"""
    a_inv = mod_inverse(a, 26)
    if a_inv is None:
        return None

    result = []
    for c in ciphertext:
        if c.isalpha():
            is_upper = c.isupper()
            y = ord(c.lower()) - ord('a')
            x = (a_inv * (y - b)) % 26
            decrypted = chr(x + ord('a'))
            result.append(decrypted.upper() if is_upper else decrypted)
        else:
            result.append(c)
    return ''.join(result)

def score_text(text):
    """
    对解密结果进行英文可读性评分
    统计常见英文单词出现次数，核心高频词额外加权
    """
    words = re.findall(r'[a-zA-Z]+', text.lower())
    score = 0

    # 基础分：匹配常见单词
    for w in words:
        if w in COMMON_WORDS:
            score += 1

    # 核心高频词额外加权（×2）
    bonus_words = ['the', 'and', 'of', 'to', 'a', 'in', 'is', 'it', 'was', 'as',
                   'his', 'had', 'for', 'but', 'not', 'be', 'this', 'have', 'from',
                   'they', 'we', 'you', 'do', 'at', 'that', 'or', 'an', 'will',
                   'my', 'all', 'would', 'there', 'their', 'what', 'so', 'up',
                   'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me',
                   'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him',
                   'know', 'take', 'see', 'come', 'over', 'think', 'also', 'back',
                   'after', 'use', 'how', 'our', 'work', 'first', 'well', 'way',
                   'even', 'new', 'want', 'any', 'these', 'give', 'day', 'most',
                   'us', 'keep', 'hope', 'dream', 'heart']
    for bw in bonus_words:
        score += words.count(bw) * 2

    return score

def brute_force(ciphertext):
    """暴力破解：遍历全部312种密钥组合，返回评分最高的结果"""
    print("\n" + "=" * 50)
    print("暴力破解（312种密钥组合）")
    print("-" * 50)

    results = []
    for a in VALID_A:
        for b in range(26):
            decrypted = affine_decrypt(ciphertext, a, b)
            if decrypted:
                score = score_text(decrypted)
                results.append((score, a, b, decrypted))

    # 按评分降序
    results.sort(key=lambda x: x[0], reverse=True)

    # 显示Top 5
    print(f"{'排名':<4} {'a':<3} {'b':<3} {'评分':<6} 前80字符预览")
    print("-" * 50)
    for i, (score, a, b, text) in enumerate(results[:5], 1):
        preview = text[:80].replace('\n', ' ')
        print(f"{i:<4} {a:<3} {b:<3} {score:<6} {preview}")

    best_score, best_a, best_b, best_text = results[0]
    print(f"\n最佳结果: a={best_a}, b={best_b}, 评分={best_score}")
    return (best_a, best_b, best_text, results[:5])


# ========== 5. 主程序 ==========

if __name__ == "__main__":
    print("仿射密码破解程序")
    print("=" * 50)
    print(f"密文:\n{CIPHERTEXT}\n")

    # 方法1：频率分析
    top_freq = frequency_analysis(CIPHERTEXT)
    key1 = derive_key_from_frequency(top_freq)

    # 方法2：暴力破解
    a_bf, b_bf, plaintext, top5 = brute_force(CIPHERTEXT)

    # 交叉验证
    print("\n" + "=" * 50)
    print("交叉验证")
    print("-" * 50)
    if key1:
        a_math, b_math = key1
        print(f"频率分析: a={a_math}, b={b_math}")
        print(f"暴力破解: a={a_bf}, b={b_bf}")
        if a_math == a_bf and b_math == b_bf:
            print("✓ 两路结果一致，密钥确认正确！")
        else:
            print("✗ 结果不一致，需人工检查")
    else:
        print("频率分析失败，以暴力破解结果为准")

    # 输出明文
    print("\n" + "=" * 50)
    print("破解结果（英文明文）")
    print("-" * 50)
    print(plaintext)
