"""
nomibench_data.py
The question bank for NomiBench, split into five categories:

    knowledge   - general knowledge
    instruction - instruction following
    code        - Python coding tasks (auto-executed & unit-tested)
    logic       - logic / reasoning puzzles
    math        - arithmetic & word problems

Each question is a plain dict consumed by nomibench_eval.evaluate().
"""

DEFAULT_MAX_TOKENS = {
    "knowledge": 100,
    "instruction": 120,
    "code": 450,
    "logic": 180,
    "math": 150,
}

CATEGORY_ORDER = ["knowledge", "instruction", "code", "logic", "math"]

CATEGORY_LABELS = {
    "knowledge": "General Knowledge",
    "instruction": "Instruction Following",
    "code": "Code",
    "logic": "Logic",
    "math": "Math",
}


# --------------------------------------------------------------------------
# General knowledge
# --------------------------------------------------------------------------

KNOWLEDGE = [
    {"id": "know_01", "type": "exact", "answer": "Canberra",
     "prompt": "What is the capital of Australia?"},
    {"id": "know_02", "type": "exact", "answer": "Au",
     "prompt": "What is the chemical symbol for gold?"},
    {"id": "know_03", "type": "exact", "answer": "Jupiter",
     "prompt": "What is the largest planet in our solar system?"},
    {"id": "know_04", "type": "contains_any", "answer": ["Orwell"],
     "prompt": "Who wrote the novel '1984'? Give the author's name."},
    {"id": "know_05", "type": "numeric", "answer": 1945,
     "prompt": "In what year did World War II end? Answer with a single year."},
    {"id": "know_06", "type": "numeric", "answer": 7,
     "prompt": "How many continents are there on Earth? Answer with a single number."},
    {"id": "know_07", "type": "exact", "answer": "Yen",
     "prompt": "What is the name of the official currency of Japan?"},
    {"id": "know_08", "type": "exact", "answer": "H2O",
     "prompt": "What is the chemical formula for water?"},
    {"id": "know_09", "type": "multiple_choice", "answer": "B",
     "prompt": "Which planet is known as the Red Planet?\nA) Venus\nB) Mars\nC) Jupiter\nD) Saturn\nAnswer with the letter of the correct option."},
    {"id": "know_10", "type": "multiple_choice", "answer": "B",
     "prompt": "Who wrote 'Romeo and Juliet'?\nA) Charles Dickens\nB) William Shakespeare\nC) Mark Twain\nD) Jane Austen\nAnswer with the letter of the correct option."},
    {"id": "know_11", "type": "numeric", "answer": 299792, "tolerance": 0.02,
     "prompt": "What is the approximate speed of light in a vacuum, in kilometers per second? Answer with a single number."},
    {"id": "know_12", "type": "numeric", "answer": 206,
     "prompt": "How many bones are there in the adult human body? Answer with a single number."},
    {"id": "know_13", "type": "multiple_choice", "answer": "C",
     "prompt": "What is the smallest prime number?\nA) 0\nB) 1\nC) 2\nD) 3\nAnswer with the letter of the correct option."},
    {"id": "know_14", "type": "exact", "answer": "Paris",
     "prompt": "What is the capital of France?"},
    {"id": "know_15", "type": "multiple_choice", "answer": "C",
     "prompt": "Which gas do plants primarily absorb from the atmosphere for photosynthesis?\nA) Oxygen\nB) Nitrogen\nC) Carbon Dioxide\nD) Hydrogen\nAnswer with the letter of the correct option."},
]


# --------------------------------------------------------------------------
# Instruction following
# --------------------------------------------------------------------------

INSTRUCTION = [
    {"id": "instr_01", "type": "format_check", "check": "exact_match",
     "params": {"target": "Confirmed"}, "max_tokens": 20,
     "prompt": "Respond with exactly the word \"Confirmed\" and nothing else."},
    {"id": "instr_02", "type": "format_check", "check": "word_count_exact",
     "params": {"n": 10}, "max_tokens": 60,
     "prompt": "Write a sentence about the ocean using exactly 10 words."},
    {"id": "instr_03", "type": "format_check", "check": "numbered_list",
     "params": {"n": 3}, "max_tokens": 60,
     "prompt": "List exactly three colors, one per line, numbered like:\n1. ...\n2. ...\n3. ..."},
    {"id": "instr_04", "type": "format_check", "check": "yes_no",
     "params": {"expected": "yes"}, "max_tokens": 20,
     "prompt": "Answer the following question with only the single word 'Yes' or 'No': Is the sky typically blue during a clear day?"},
    {"id": "instr_05", "type": "format_check", "check": "all_caps",
     "params": {}, "max_tokens": 60,
     "prompt": "Write your response in all uppercase letters: describe a cat in one sentence."},
    {"id": "instr_06", "type": "format_check", "check": "no_letter",
     "params": {"letter": "e"}, "max_tokens": 60,
     "prompt": "Do not use the letter 'e' anywhere in your response. Describe a dog in one sentence."},
    {"id": "instr_07", "type": "format_check", "check": "valid_json",
     "params": {"expected": {"name": "Alex", "age": 30}}, "max_tokens": 60,
     "prompt": "Output valid JSON (and nothing else) with keys \"name\" and \"age\", where name is \"Alex\" and age is 30."},
    {"id": "instr_08", "type": "format_check", "check": "starts_with",
     "params": {"prefix": "Certainly"}, "max_tokens": 60,
     "prompt": "Start your response with the word \"Certainly\", then briefly explain what a rainbow is."},
    {"id": "instr_09", "type": "format_check", "check": "single_word_contains",
     "params": {"word": "cold"}, "max_tokens": 20,
     "prompt": "What is the opposite of \"hot\"? Provide your answer as exactly one word."},
    {"id": "instr_10", "type": "format_check", "check": "sentence_count",
     "params": {"n": 2}, "max_tokens": 100,
     "prompt": "Write exactly two sentences about space exploration."},
    {"id": "instr_11", "type": "format_check", "check": "exact_match",
     "params": {"target": "The quick brown fox jumps."}, "max_tokens": 30,
     "prompt": "Repeat the following phrase exactly, with no other text: \"The quick brown fox jumps.\""},
    {"id": "instr_12", "type": "format_check", "check": "ends_with",
     "params": {"suffix": "Done"}, "max_tokens": 60,
     "prompt": "Briefly describe how to make a sandwich, and end your response with the exact word \"Done\"."},
    {"id": "instr_13", "type": "format_check", "check": "exact_match",
     "params": {"target": "7"}, "max_tokens": 20,
     "prompt": "Respond only with a single number between 1 and 10, nothing else. Choose the number 7."},
    {"id": "instr_14", "type": "format_check", "check": "line_count",
     "params": {"n": 3}, "max_tokens": 60,
     "prompt": "Write a short haiku about autumn. Respond with exactly 3 lines and nothing else (no title, no explanation)."},
    {"id": "instr_15", "type": "format_check", "check": "single_word_contains",
     "params": {"word": "bonjour"}, "max_tokens": 20,
     "prompt": "Translate the word \"hello\" into French. Respond with only the translated word."},
]


# --------------------------------------------------------------------------
# Code (each is auto-executed and unit-tested)
# --------------------------------------------------------------------------

CODE = [
    {"id": "code_01", "type": "code", "function_name": "add",
     "tests": [{"args": [2, 3], "expected": 5}, {"args": [-1, 1], "expected": 0}, {"args": [0, 0], "expected": 0}],
     "prompt": "Write a Python function named `add` that takes two numbers `a` and `b` and returns their sum. Return only a Python code block."},
    {"id": "code_02", "type": "code", "function_name": "is_palindrome",
     "tests": [{"args": ["racecar"], "expected": True}, {"args": ["hello"], "expected": False}, {"args": ["a"], "expected": True}],
     "prompt": "Write a Python function named `is_palindrome` that takes a string `s` and returns True if it reads the same forwards and backwards, otherwise False. Return only a Python code block."},
    {"id": "code_03", "type": "code", "function_name": "factorial",
     "tests": [{"args": [5], "expected": 120}, {"args": [0], "expected": 1}, {"args": [3], "expected": 6}],
     "prompt": "Write a Python function named `factorial` that takes a non-negative integer `n` and returns n! (n factorial). Return only a Python code block."},
    {"id": "code_04", "type": "code", "function_name": "fibonacci",
     "tests": [{"args": [0], "expected": 0}, {"args": [1], "expected": 1}, {"args": [10], "expected": 55}],
     "prompt": "Write a Python function named `fibonacci` that takes an integer `n` and returns the nth Fibonacci number, 0-indexed (fibonacci(0)=0, fibonacci(1)=1). Return only a Python code block."},
    {"id": "code_05", "type": "code", "function_name": "reverse_string",
     "tests": [{"args": ["abc"], "expected": "cba"}, {"args": ["hello"], "expected": "olleh"}],
     "prompt": "Write a Python function named `reverse_string` that takes a string `s` and returns it reversed. Return only a Python code block."},
    {"id": "code_06", "type": "code", "function_name": "is_prime",
     "tests": [{"args": [7], "expected": True}, {"args": [8], "expected": False}, {"args": [1], "expected": False}, {"args": [2], "expected": True}],
     "prompt": "Write a Python function named `is_prime` that takes an integer `n` and returns True if it is a prime number, otherwise False. Return only a Python code block."},
    {"id": "code_07", "type": "code", "function_name": "count_vowels",
     "tests": [{"args": ["hello"], "expected": 2}, {"args": ["xyz"], "expected": 0}, {"args": ["AEIOU"], "expected": 5}],
     "prompt": "Write a Python function named `count_vowels` that takes a string `s` and returns the number of vowels (a, e, i, o, u), case-insensitive. Return only a Python code block."},
    {"id": "code_08", "type": "code", "function_name": "max_of_list",
     "tests": [{"args": [[3, 1, 4, 1, 5, 9, 2, 6]], "expected": 9}, {"args": [[-5, -1, -10]], "expected": -1}],
     "prompt": "Write a Python function named `max_of_list` that takes a list of numbers `lst` and returns the largest value. Return only a Python code block."},
    {"id": "code_09", "type": "code", "function_name": "fizzbuzz",
     "tests": [{"args": [15], "expected": [1, 2, "Fizz", 4, "Buzz", "Fizz", 7, 8, "Fizz", "Buzz", 11, "Fizz", 13, 14, "FizzBuzz"]}],
     "prompt": "Write a Python function named `fizzbuzz` that takes an integer `n` and returns a list of length n for the numbers 1 to n inclusive, where multiples of 3 are replaced with \"Fizz\", multiples of 5 with \"Buzz\", and multiples of both with \"FizzBuzz\". Return only a Python code block."},
    {"id": "code_10", "type": "code", "function_name": "sum_of_digits",
     "tests": [{"args": [1234], "expected": 10}, {"args": [0], "expected": 0}, {"args": [999], "expected": 27}],
     "prompt": "Write a Python function named `sum_of_digits` that takes a non-negative integer `n` and returns the sum of its digits. Return only a Python code block."},
    {"id": "code_11", "type": "code", "function_name": "is_anagram",
     "tests": [{"args": ["listen", "silent"], "expected": True}, {"args": ["hello", "world"], "expected": False}, {"args": ["Dormitory", "Dirty Room"], "expected": True}],
     "prompt": "Write a Python function named `is_anagram` that takes two strings `a` and `b` and returns True if they are anagrams of each other, ignoring case and spaces. Return only a Python code block."},
    {"id": "code_12", "type": "code", "function_name": "binary_search",
     "tests": [{"args": [[1, 3, 5, 7, 9, 11], 7], "expected": 3}, {"args": [[1, 3, 5, 7, 9, 11], 4], "expected": -1}, {"args": [[], 5], "expected": -1}],
     "prompt": "Write a Python function named `binary_search` that takes a sorted list `lst` and a value `target`, and returns the index of `target` in the list using binary search, or -1 if not found. Return only a Python code block."},
    {"id": "code_13", "type": "code", "function_name": "gcd",
     "tests": [{"args": [12, 18], "expected": 6}, {"args": [7, 13], "expected": 1}, {"args": [0, 5], "expected": 5}],
     "prompt": "Write a Python function named `gcd` that takes two non-negative integers `a` and `b` and returns their greatest common divisor. Return only a Python code block."},
    {"id": "code_14", "type": "code", "function_name": "flatten_list",
     "tests": [{"args": [[[1, 2], [3], [4, 5]]], "expected": [1, 2, 3, 4, 5]}, {"args": [[[1], [], [2, 3, 4]]], "expected": [1, 2, 3, 4]}],
     "prompt": "Write a Python function named `flatten_list` that takes a list of lists `nested` and returns a single flattened list (one level deep), preserving order. Return only a Python code block."},
    {"id": "code_15", "type": "code", "function_name": "remove_duplicates",
     "tests": [{"args": [[1, 2, 2, 3, 1]], "expected": [1, 2, 3]}, {"args": [["a", "b", "a", "c"]], "expected": ["a", "b", "c"]}],
     "prompt": "Write a Python function named `remove_duplicates` that takes a list `lst` and returns a new list with duplicates removed, preserving the original order of first occurrence. Return only a Python code block."},
]


# --------------------------------------------------------------------------
# Logic
# --------------------------------------------------------------------------

LOGIC = [
    {"id": "logic_01", "type": "yes_no", "answer": "yes",
     "prompt": "If all Bloops are Razzies, and all Razzies are Lazzies, are all Bloops definitely Lazzies? Answer with only 'Yes' or 'No'."},
    {"id": "logic_02", "type": "numeric", "answer": 32,
     "prompt": "What comes next in the sequence 2, 4, 8, 16, __? Answer with a single number."},
    {"id": "logic_03", "type": "single_letter", "answer": "C",
     "prompt": "A is taller than B. B is taller than C. Who is the shortest? Answer with a single letter (A, B, or C)."},
    {"id": "logic_04", "type": "contains_any", "answer": ["Friday"],
     "prompt": "If today is Wednesday, what day of the week will it be in 100 days? Answer with the day name."},
    {"id": "logic_05", "type": "numeric", "answer": 9,
     "prompt": "Which number does not belong in this list, and why: 2, 3, 5, 9, 7? State the odd one out as a single number at the end of your response."},
    {"id": "logic_06", "type": "numeric", "answer": 3,
     "prompt": "There are 5 apples on a table and you take away 3 of them. How many apples do you have? Answer with a single number."},
    {"id": "logic_07", "type": "numeric", "answer": 9,
     "prompt": "A farmer has 17 sheep. All but 9 die. How many sheep are left? Answer with a single number."},
    {"id": "logic_08", "type": "contains_any", "answer": ["carrot"],
     "prompt": "Which word is the odd one out: apple, banana, carrot, orange? Answer with just the word."},
    {"id": "logic_09", "type": "contains_any", "answer": ["sock", "shoe", "socks", "shoes"],
     "prompt": "Complete the analogy: Hand is to Glove as Foot is to ____."},
    {"id": "logic_10", "type": "numeric", "answer": 1,
     "prompt": "Two trucks are 100 km apart, moving toward each other at 40 km/h and 60 km/h respectively. In how many hours will they meet? Answer with a single number."},
    {"id": "logic_11", "type": "contains_any", "answer": ["second"],
     "prompt": "In a race, you overtake the person currently in second place. What position are you in now? Answer with a single word (an ordinal, e.g. 'First')."},
    {"id": "logic_12", "type": "numeric", "answer": 5,
     "prompt": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? Answer in minutes as a single number."},
    {"id": "logic_13", "type": "yes_no", "answer": "no",
     "prompt": "All roses are flowers. Some flowers fade quickly. Can we logically conclude that some roses fade quickly? Answer with only 'Yes' or 'No'."},
    {"id": "logic_14", "type": "single_letter", "answer": "I",
     "prompt": "What is the next letter in this pattern: A, C, E, G, __? Answer with a single letter."},
    {"id": "logic_15", "type": "numeric", "answer": 7.5, "tolerance": 0.1,
     "prompt": "A clock shows 3:15. What is the angle between the hour and minute hands, in degrees? Answer with a single number."},
]


# --------------------------------------------------------------------------
# Math
# --------------------------------------------------------------------------

MATH = [
    {"id": "math_01", "type": "numeric", "answer": 391,
     "prompt": "What is 17 multiplied by 23? Answer with a single number."},
    {"id": "math_02", "type": "numeric", "answer": 12,
     "prompt": "What is the square root of 144? Answer with a single number."},
    {"id": "math_03", "type": "numeric", "answer": 36,
     "prompt": "What is 15% of 240? Answer with a single number."},
    {"id": "math_04", "type": "numeric", "answer": 6,
     "prompt": "Solve for x: 2x + 5 = 17. What is x? Answer with a single number."},
    {"id": "math_05", "type": "numeric", "answer": 153.94, "tolerance": 0.02,
     "prompt": "What is the area of a circle with radius 7? Use pi \u2248 3.14159 and answer with a single number, rounded to two decimal places."},
    {"id": "math_06", "type": "numeric", "answer": 5040,
     "prompt": "What is 7 factorial (7!)? Answer with a single number."},
    {"id": "math_07", "type": "numeric", "answer": 18,
     "prompt": "What is the average (mean) of the numbers 4, 8, 15, 16, 23, and 42? Answer with a single number."},
    {"id": "math_08", "type": "exact", "answer": "3/4",
     "prompt": "Convert the decimal 0.75 to a simplified fraction. Answer in the form a/b."},
    {"id": "math_09", "type": "numeric", "answer": 100,
     "prompt": "If you invest $1000 at 5% simple annual interest, how much interest (in dollars) do you earn after 2 years? Answer with a single number."},
    {"id": "math_10", "type": "numeric", "answer": 26,
     "prompt": "What is the perimeter of a rectangle with length 8 and width 5? Answer with a single number."},
    {"id": "math_11", "type": "numeric", "answer": 1024,
     "prompt": "What is 2 raised to the power of 10 (2^10)? Answer with a single number."},
    {"id": "math_12", "type": "numeric", "answer": 12,
     "prompt": "What is the least common multiple (LCM) of 4 and 6? Answer with a single number."},
    {"id": "math_13", "type": "numeric", "answer": 40,
     "prompt": "A train travels 60 miles in 1.5 hours at a constant speed. What is its speed in miles per hour? Answer with a single number."},
    {"id": "math_14", "type": "numeric", "answer": 55,
     "prompt": "What is the sum of the first 10 natural numbers (1 through 10)? Answer with a single number."},
    {"id": "math_15", "type": "exact", "answer": "1/6",
     "prompt": "What is the probability of rolling a 6 on a single roll of a fair six-sided die? Answer as a simplified fraction."},
]


QUESTION_BANK = {
    "knowledge": KNOWLEDGE,
    "instruction": INSTRUCTION,
    "code": CODE,
    "logic": LOGIC,
    "math": MATH,
}
