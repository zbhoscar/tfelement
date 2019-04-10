import sys, time
import multiprocessing

def main():
    nnnn = 5
    p = multiprocessing.Pool(nnnn)
    for i in range(nnnn):
        sys.stdout.write('\n')
        p.apply_async(whynot, args=(i,5))
    p.close()
    p.join()

def whynot(n,r):
    # print('abc%d' % n)
    # return None
    for i in range(5):  

        sys.stdout.write(' '* 10 + '\r')  
        sys.stdout.flush()  
        sys.stdout.write(str(i) * (5 - i) + '\r')  
        sys.stdout.flush()  
        time.sleep(1)  



if __name__ == '__main__':
    main()

# for i in range(5):  
#     sys.stdout.write(' ' * 10 + '\r')  
#     sys.stdout.flush()  
#     print( i  )
#     sys.stdout.write(str(i) * (5 - i) + '\r')  
#     sys.stdout.flush()  
#     time.sleep(1)  