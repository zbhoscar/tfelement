import multiprocessing

split_list = [1,2,3,4,5,6,7,8,9]

def list_to_placeholder(inn, num):
    ott = [inn*2, inn*num]
    return ott
    
# multiprocessing.freeze_support()
p = multiprocessing.Pool()

result = []
out1 = []
out2 = []
# print('Checking naive files:', end='\n    ')
for em in split_list:
    result.append(p.apply_async(list_to_placeholder, args=(em,10)))
p.close()
p.join()

[res.get() for res in result]

[out1, out2] = result.get()

# tasks = split_list
# res = p.apply_async(list_to_placeholder, ((i) for i in tasks))
# p.close()
# print(res.get())
# print(result[:].get())
# out1 = []
# out2 = []
# for res in result:
#     out1.extend(res.get()[0])
#     out2.extend(res.get()[1])
