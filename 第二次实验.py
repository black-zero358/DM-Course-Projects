# 利用range函数产生一个包含整数0到20的列表
number_list = list(range(21))

# 创建一个字典，将偶数保存在该字典的第一个键值对中，奇数保存在第二个键值对中
even_numbers = [num for num in number_list if num % 2 == 0]
odd_numbers = [num for num in number_list if num % 2 != 0]
result_dict = {'even': even_numbers, 'odd': odd_numbers}
print(result_dict)

# 编写程序，计算并输出300以内最大的素数
def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

max_prime = 0
for number in range(300, 1, -1):
    if is_prime(number):
        max_prime = number
        break
print("300以内最大的素数是:", max_prime)


# 编写程序，计算各位数字都不相同的所有三位数的个数，并输出最大的10个
unique_numbers_count = 0
unique_numbers_list = []

for i in range(100, 1000):
    digits = [int(digit) for digit in str(i)]
    if len(set(digits)) == 3:
        unique_numbers_count += 1
        unique_numbers_list.append(i)

print("各位数字都不相同的所有三位数的个数:", unique_numbers_count)
print("最大的10个数字:", sorted(unique_numbers_list)[-10:])

# 利用递归获取斐波那契数列中的第12个数，并将该值返回给调用者
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

fib_12 = fibonacci(12)
print("斐波那契数列中的第12个数是:", fib_12)

# 创建一个文件test.txt, 文件第一行是自己的名字，第二行是班级，第三行是学号。
with open("test.txt", "w") as file:
    file.write("Your Name\n")
    file.write("Your Class\n")
    file.write("Your Student ID\n")

# 分别用read, readline,readlines读取test.txt的文件内容。
with open("test.txt", "r") as file:
    content = file.read()
    print("使用read读取文件内容:\n", content)

with open("test.txt", "r") as file:
    content = file.readline()
    print("使用readline读取文件内容:\n", content)

with open("test.txt", "r") as file:
    content = file.readlines()
    print("使用readlines读取文件内容:\n", content)

import random

def guess_number(secret_number, max_attempts):
    print("猜数字游戏开始！")
    print(f"你有 {max_attempts} 次猜测的机会。")

    attempts = 0
    while attempts < max_attempts:
        guess = int(input("请输入你猜测的数字："))

        if guess < secret_number:
            print("太小了，请再试一次。")
        elif guess > secret_number:
            print("太大了，请再试一次。")
        else:
            print("恭喜你，猜对了！")
            return True

        attempts += 1

    print(f"很遗憾，你已经用完了所有的猜测机会。正确的数字是 {secret_number}。")
    return False

def main():
    # 设置猜测范围和最大猜测次数
    min_number = 1
    max_number = 100
    max_attempts = 5

    # 随机生成一个待猜测的数字
    secret_number = random.randint(min_number, max_number)

    # 开始游戏
    guess_number(secret_number, max_attempts)

if __name__ == "__main__":
    main()
