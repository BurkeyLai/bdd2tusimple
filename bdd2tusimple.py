import json
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# Load json file
with open('bdd100k_labels_images_train.json') as json_file:
    data = json.load(json_file)

def bezier2poly(V, T):
    num_vertices = len(V)
    P = []
    vertices_L = []
    vertices_C = []
    num_L = 0
    num_C = 0
    order = 1
    A = []
    B = []
    C = []
    X = []
    Y = []
    
    for tdx, t in enumerate(T):
        if t == 'L':
            num_L = num_L + 1
            vertices_L.append(V[tdx])
        elif t == 'C':
            num_C = num_C + 1
            vertices_C.append(V[tdx])
    vertices_L = np.array(vertices_L)
    vertices_C = np.array(vertices_C)
    
    if num_C == 0:
        order = 1
        
        A = np.array([[vertices_L[0, 0], 1],
                      [vertices_L[1, 0], 1]])
        if np.linalg.det(A) != 0:
            B = np.array([vertices_L[0, 1], vertices_L[1, 1]]).reshape(2, 1)
            A_inv = np.linalg.inv(A)
            C = A_inv.dot(B)
        else:
            A = np.array([[vertices_L[0, 0] + 1, 1],
                      [vertices_L[1, 0], 1]])
            B = np.array([vertices_L[0, 1], vertices_L[1, 1]]).reshape(2, 1)
            A_inv = np.linalg.inv(A)
            C = A_inv.dot(B)
            
    elif num_C == 3:
        order = num_C - 1
        ### 取得參數式的係數。ex:
        
        for j in range(order + 1):
            p = 0
            for i in range(j + 1):
                #print(np.array(vertices_C[i]))
                p = p + (math.pow(-1, i + j) * np.array(vertices_C[i])) / (math.factorial(i) * math.factorial(j - i))
            P.append((math.factorial(order) / math.factorial(order - j)) * p)
        P = np.array(P)
        
        
        ### 帶入任意t值求得x, y座標
        answer = ((P[2, 1] / P[2, 0]) * vertices_L[0, 0] - vertices_L[0, 1] + P[0, 1] - (P[2, 1] / P[2, 0]) * P[0, 0]) \
            / ((P[2, 1] / P[2, 0]) * P[1, 0] - P[1, 1])
        
        for idx in [answer - 1, answer, answer + 1]:
            X.append(P[2, 0] * math.pow(idx, 2) + P[1, 0] * idx + P[0, 0])
            Y.append(P[2, 1] * math.pow(idx, 2) + P[1, 1] * idx + P[0, 1])
        
        X = np.array(X)
        Y = np.array(Y)
        
        ### 求聯立方程式
        A = np.array([[math.pow(X[0], 2), X[0], 1],
                      [math.pow(X[1], 2), X[1], 1],
                      [math.pow(X[2], 2), X[2], 1]])
        #print(np.linalg.det(A / 1000))
        if np.linalg.det(A) != 0:
            B = np.array([Y[0], Y[1], Y[2]]).reshape(3, 1)
            A_inv = np.linalg.inv(A)
            C = A_inv.dot(B)
        else:
            C = np.array([[0], [0], [0]])
        
    return C, order

bdd_file_name = []
with open('../../../label_data_bdd100k.json', 'w') as f:
    for i in data:
        lane_lines = []
        final_lane_lines = []
        vertice_y = []

        print(i['name'])

        category = []
        for j in i['labels']:
            category.append(j['category'])

            lines_vertices = []
            lines_types = []
            if j['category'] == 'lane' and len(j['poly2d']) == 1:
                lines_vertices = j['poly2d'][0]['vertices']
                lines_types = j['poly2d'][0]['types']
                for v in lines_vertices:
                    vertice_y.append(v[1])
                num_C = 0
                for t in lines_types:
                    if t == 'C':
                        num_C = num_C + 1

                if num_C <= 3:
                    poly, order = bezier2poly(lines_vertices, lines_types)
                else:
                    continue

                x = []
                y = np.arange(160, 720, 10)
                if order == 1:
                    x = (y - poly[1]) / poly[0]

                elif order == 2:

                    a = poly[0, 0]
                    b = poly[1, 0]
                    c = poly[2, 0] - y
                    for cc in c:
                        if (b * b - 4 * a * cc) >= 0:
                            Delte = math.sqrt(b * b - 4 * a * cc)
                            if (- b - Delte) * (2 * a) >= 0:
                                x.append((- b - Delte) / (2 * a))
                            elif (- b + Delte) * (2 * a) >= 0:
                                x.append((- b + Delte) / (2 * a))
                            else:
                                x.append(-2)
                        else:
                            x.append(-2)

                line = np.stack((x, y), axis = 1)
                ### Delete those out of image's shape
                toDelete = []
                for pdx, point in enumerate(line):
                    if point[0] < -2 or point[0] > 1280:
                        toDelete.append(pdx)
                line = np.delete(line, toDelete, 0)
                lane_lines.append(line)


        ### 刪除空陣列
        delete_lines = []
        for l in range(len(lane_lines)):
            if len(lane_lines[l]) == 0:
                delete_lines.append(l)
        lane_lines = np.delete(lane_lines, delete_lines, 0)

        ### 排序 bubble sort
        for p in range(len(lane_lines) - 1):
            for pp in range(len(lane_lines) - 1 - p):
                if (lane_lines[pp][-1, 1] > lane_lines[pp + 1][-1, 1]) or \
                    (lane_lines[pp][-1, 0] > lane_lines[pp + 1][-1, 0] and lane_lines[pp][-1, 1] >= lane_lines[pp + 1][-1, 1]):
                    lane_lines[pp], lane_lines[pp + 1] = lane_lines[pp + 1], lane_lines[pp]

        #print(lane_lines)
        ### 刪除那些屬於同一條線的多餘的data
        delete_lines = []
        for l in range(len(lane_lines) - 1):
            this_last_point = lane_lines[l][-1]
            next_last_point = lane_lines[l + 1][-1]
            #d = math.sqrt(pow(this_last_point[0] - next_last_point[0], 2) + pow(this_last_point[1] - next_last_point[1], 2))
            x_d = abs(this_last_point[0] - next_last_point[0])
            y_d = abs(this_last_point[1] - next_last_point[1])
            if x_d > 200:
                continue
            elif x_d <= 200 and y_d > 100:
                continue
            else:
                delete_lines.append(l)
        lane_lines = np.delete(lane_lines, delete_lines, 0)


        ### 修改那些太超出的點為-2

        for ldx, l in enumerate(lane_lines):
            for pdx, p in enumerate(l):
                if p[1] < min(vertice_y):
                    p[0] = -2

        ### 補足那些點不夠的線到長度為56
        for ldx, l in enumerate(lane_lines):
            if len(l) != 56:
                point_y = l[:, 1]
                for h in range(0, 56):
                    if (160 + h * 10) not in point_y:
                        l = np.concatenate((l, np.array([[-2, 160 + h * 10]])), axis = 0)

                #print(l)
                final_lane_lines.append(l)

        ### 排序 bubble sort
        for l in final_lane_lines:
            for p in range(len(l) - 1):
                for pp in range(len(l) - 1 - p):
                    if l[pp, 1] > l[pp + 1, 1]:
                        l[pp, 0], l[pp + 1, 0] = l[pp + 1, 0], l[pp, 0]
                        l[pp, 1], l[pp + 1, 1] = l[pp + 1, 1], l[pp, 1]
        
        ### 寫入json file
        if len(final_lane_lines) <= 4 and len(final_lane_lines) > 0:
            lane_data = {"lanes": [], "h_samples": [], "raw_file": ""}
            for l in final_lane_lines:
                lane_data['lanes'].append(l[:, 0])
            lane_data['lanes'] = (np.array(lane_data['lanes']).astype(int)).tolist()
            lane_data["h_samples"] = final_lane_lines[0][:, 1]
            lane_data["h_samples"] = (np.array(lane_data["h_samples"]).astype(int)).tolist()
            lane_data["raw_file"] = "bdd100k_images/bdd100k/images/100k/train/" + i['name']
            json.dump(lane_data, f)
            f.write('\n')
            
        elif len(final_lane_lines) > 4:
            lane_data = {"lanes": [], "h_samples": [], "raw_file": ""}
            lane_id = []
            
            for ldx, l in enumerate(final_lane_lines):
                if l[-1, 0] != -2 and l[-1, 1] == 710 and len(lane_id) < 4:
                    lane_id.append(ldx)
            if len(lane_id) < 4 and len(lane_id) > 0:
                lane_id.extend(random.sample([c for c in \
                                              [cc for cc in range(len(final_lane_lines))] if not c in lane_id], 4 - len(lane_id)))
            
            if len(lane_id) == 0:
                lane_id = random.sample([cc for cc in range(len(final_lane_lines))], 4)
                
            lane_id = sorted(lane_id)
            for l in lane_id:
                lane_data['lanes'].append(final_lane_lines[l][:, 0])
            lane_data['lanes'] = (np.array(lane_data['lanes']).astype(int)).tolist()
            lane_data["h_samples"] = final_lane_lines[0][:, 1]
            lane_data["h_samples"] = (np.array(lane_data["h_samples"]).astype(int)).tolist()
            lane_data["raw_file"] = "bdd100k_images/bdd100k/images/100k/train/" + i['name']
            json.dump(lane_data, f)
            f.write('\n')

        if 'lane' in category:
            #print(category)
            bdd_file_name.append(i['name'])