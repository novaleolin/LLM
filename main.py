import csv
import queue
import time
import threading
import numpy as np


JOB_NUM = 100  # 发送请求的个数
arrival_rate = 1 / 10  # 用户请求到达速率，每秒10个请求
quantum = 1  # Quantum大小
quantum_rate = 2  # Quantum比率
queue_num = 3  # 队列数量
DATASET_PATH = 'orca_set.csv'  # 数据集文件路径
num_jct = {key: None for key in range(1, 101)}

# 在opt-1.3B上的实验数据 单位: ms
x = [1, 4, 16, 64, 256, 512, 1024]
first_time = [5.88, 5.93, 6.57, 8.04, 23.8, 43.9, 98.5]
next_time = [5.13, 5.11, 5.16, 5.22, 5.52, 5.72, 5.82]

# 通过实验数据拟合每次迭代推理时间
z1 = np.polyfit(x, first_time, 1)
p1 = np.poly1d(z1)

z2 = np.polyfit(x, next_time, 1)
p2 = np.poly1d(z2)


def fit_first_iter_time(prompt_length):
    return p1(prompt_length)


def fit_next_iter_time(prompt_length):
    return p2(prompt_length)


class Request:  # 推理请求，理论上输出长度未知，但为仿真实验，需要事先确定
    def __init__(self, j_id, prompt_length, output_length):
        self.j_id = j_id
        self.prompt_length = int(prompt_length)
        self.output_length = int(output_length)
        self.first_iter_time = fit_first_iter_time(prompt_length)
        self.next_iter_time = fit_next_iter_time(prompt_length)
        self.iter_count = 0  # 请求执行了几次迭代，iter_count==output_length时完成整个推理
        self.priority = -1  # 请求目前处于第几级队列
        self.create_time = time.time()  # 请求创建时间
        self.token_number = 0


# 用户请求生成线程
class RequestGenerator(threading.Thread):
    def __init__(self, arrival_rate, request_queue):
        super().__init__()
        self.arrival_rate = arrival_rate  # arrival rate = 1s / job interval
        self.request_queue = request_queue
        self.count = 0  # 用于跟踪已发送的请求数量

    def run(self):
        prompt_length_list = []
        output_length_list = []

        # 此处为读取orca数据集中的数据来构造request，可自行修改路径
        f = open(DATASET_PATH, 'r')
        with f:
            reader = csv.reader(f)
            for row in reader:
                if self.count == 0:
                    self.count += 1
                    continue

                prompt_length_list.append(row[0])
                output_length_list.append(row[1])

        j_id = 0
        while j_id < JOB_NUM:
            output_ = output_length_list[j_id]
            input_ = prompt_length_list[j_id]
            request = Request(j_id, input_, output_)
            self.request_queue.put(request)
            j_id += 1  # 任务请求编号
            time.sleep(1 / self.arrival_rate)


def log_token_generation(self):
    # 输出生成的token信息
    print(f"Job {self.j_id} - Generated token {self.token_number}")

def print_average_jct(schedule):
    # 输出平均JCT
  #  for i in range(0, len(schedule)):
   #     print(f"J_ID: {} JCT:{} seconds.")
    average_jct = sum(schedule.ave_jct) / len(schedule.ave_jct)
    print(f"Average JCT: {average_jct} seconds.")


class SkipJoinMLFQScheduler:

    def __init__(self, first_quantum=6, quantum_rate=4, queue_num=4):
        # super().__init__()
        self.quantum_list = []
        self.multi_level_priority_queue = []
        self.executed = 0  # 已经完成的请求数量


        # first quantum/Q1 is the min iteration time
        for i in range(queue_num):
            self.quantum_list.append(quantum_rate ** i)
            temp_q = queue.Queue(-1)
            self.multi_level_priority_queue.append(temp_q)

        self.ave_jct = []

    def getNewRequest(self, request):
        # Todo: 处理缓冲区中新到达的request，根据他们的输入长度放入多级队列中
        for i, quantum in enumerate(self.quantum_list):
            if request.first_iter_time < quantum:
                self.multi_level_priority_queue[i].put(request)
                break

    def demoteRequest(self, job):
        # Todo: 将完成了推理但还没生成完毕的请求放入下一级队列
        current_queue_index = self.quantum_list.index(job.priority)
        if current_queue_index < len(self.quantum_list) - 1:
            job.priority = self.quantum_list[current_queue_index + 1]
            self.multi_level_priority_queue[current_queue_index + 1].put(job)

    def getInferenceJob(self):
        # Todo: 返回在最高优先级的队列中的队首请求
        return self.multi_level_priority_queue[0].get() if not self.multi_level_priority_queue[0].empty() else None


def simulate_forward(iteration_time, job):
    iteration_num = scheduler.quantum_list[job.priority]  # 获取当前任务在这次推理中需要执行多少轮

    if iteration_num >= job.output_length - job.iter_count:
        iteration_num = job.output_length - job.iter_count

        for i in range(iteration_num):
            time.sleep(iteration_time / 1000)  # ms
            job.iter_count += 1
            job.token_number += 1

        jct = time.time() - job.create_time
        scheduler.ave_jct.append(jct)
        job.jct = jct

        scheduler.executed += 1
        log_token_generation(job)  # 第一个输出任务，序号+token_number
        for key, value in num_jct.items():
            if key == job.j_id:
                num_jct[key] = value

    else:
        for i in range(iteration_num):
            time.sleep(iteration_time / 1000)  # ms
            job.iter_count += 1
        log_token_generation(job)  # 第一个输出任务，序号+token_number
        scheduler.demoteRequest(job)


# 推理线程
def run(schedule, request_queue, thread_pool=None, ):
    while schedule.executed != JOB_NUM:
        # 处理请求队列中的所有请求
        print(f"request_queue.qsize:{request_queue.qsize()}.")
        for i in range(request_queue.qsize()):
            req = request_queue.get()
            scheduler.getNewRequest(req)
        # 获取并处理推理任务
        job = schedule.getInferenceJob()
        if job.iter_count == 0:
            iter_time = job.first_iter_time
        else:
            iter_time = job.next_iter_time

        args = [iter_time, job, schedule]
        # 调用模拟推理线程
        temp_thread = thread_pool.submit(lambda p: simulate_forward(*p), args)


if __name__ == '__main__':
    request_queue = queue.Queue
    # 定义并启动发送请求的用户线程
    generator = RequestGenerator(arrival_rate, request_queue)
    generator.start()

    # 定义并启动调度器线程
    scheduler = SkipJoinMLFQScheduler(first_quantum=quantum,
                                      quantum_rate=quantum_rate,
                                      queue_num=queue_num)

    run(scheduler, request_queue)

    print_average_jct(scheduler)
