'''
mean_bmi = np.mean(ex1()['bmi'])
mean_ch = np.mean(ex1()['charges'])
plt.show()
def gen(data_bmi=ex1()['bmi'], data_ch=ex1()['charges'], is_sample=True, mean_bmi=mean_bmi, mean_ch=mean_ch):
    diff_bmi = [(v - mean_bmi) for v in data_bmi]
    diff_ch = [(w - mean_ch) for w in data_ch]
    sqr_diff_bmi = [d ** 2 for d in diff_bmi]
    sqr_diff_ch = [t ** 2 for t in diff_ch]
    sum_sqr_diff_bmi = sum(sqr_diff_bmi)
    sum_sqr_diff_ch = sum(sqr_diff_ch)
    if is_sample == True:
        variance_bmi = sum_sqr_diff_bmi / (len(data_bmi) - 1)
        variance_ch = sum_sqr_diff_ch / (len(data_ch) - 1)
    else:
        variance_bmi = sum_sqr_diff_bmi / (len(data_bmi))
        variance_ch = sum_sqr_diff_ch / (len(data_ch))
    return [variance_bmi, variance_ch]
print(gen())
'''
