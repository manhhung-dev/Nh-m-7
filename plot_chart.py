import matplotlib.pyplot as plt

# Dữ liệu
labels = ['Congested (Tắc đường)', 'Normal (Bình thường)']
values = [9, 6]

# Vẽ biểu đồ tròn
plt.figure()
plt.pie(values, labels=labels, autopct='%1.1f%%')

plt.title('Kết quả dự đoán trên 15 ảnh test ngoài')

# Lưu ảnh
plt.savefig('bieu_do_test.png')

plt.show()