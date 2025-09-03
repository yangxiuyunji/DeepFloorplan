# 对角方位的处理（东北、西北、东南、西南） - 修复代码
                if direction == "东北":
                    if north_angle == 90:  # 标准朝向：右上角是东北
                        text_x = grid_max_x + margin
                        text_y = grid_min_y - text_h - margin
                    elif north_angle == 270:  # 上南下北：左下角是东北
                        text_x = grid_min_x - text_w - margin
                        text_y = grid_max_y + margin
                    elif north_angle == 0:  # 右北左南：右上角是东北
                        text_x = grid_max_x + margin
                        text_y = grid_min_y - text_h - margin
                    elif north_angle == 180:  # 左北右南：左下角是东北
                        text_x = grid_min_x - text_w - margin
                        text_y = grid_max_y + margin
                    else:
                        text_x = center_x - text_w / 2
                        text_y = center_y - text_h / 2
                elif direction == "西北":
                    if north_angle == 90:  # 标准朝向：左上角是西北
                        text_x = grid_min_x - text_w - margin
                        text_y = grid_min_y - text_h - margin
                    elif north_angle == 270:  # 上南下北：右下角是西北
                        text_x = grid_max_x + margin
                        text_y = grid_max_y + margin
                    elif north_angle == 0:  # 右北左南：右下角是西北
                        text_x = grid_max_x + margin
                        text_y = grid_max_y + margin
                    elif north_angle == 180:  # 左北右南：左上角是西北
                        text_x = grid_min_x - text_w - margin
                        text_y = grid_min_y - text_h - margin
                    else:
                        text_x = center_x - text_w / 2
                        text_y = center_y - text_h / 2
                elif direction == "东南":
                    if north_angle == 90:  # 标准朝向：右下角是东南
                        text_x = grid_max_x + margin
                        text_y = grid_max_y + margin
                    elif north_angle == 270:  # 上南下北：左上角是东南
                        text_x = grid_min_x - text_w - margin
                        text_y = grid_min_y - text_h - margin
                    elif north_angle == 0:  # 右北左南：左上角是东南
                        text_x = grid_min_x - text_w - margin
                        text_y = grid_min_y - text_h - margin
                    elif north_angle == 180:  # 左北右南：右下角是东南
                        text_x = grid_max_x + margin
                        text_y = grid_max_y + margin
                    else:
                        text_x = center_x - text_w / 2
                        text_y = center_y - text_h / 2
                elif direction == "西南":
                    if north_angle == 90:  # 标准朝向：左下角是西南
                        text_x = grid_min_x - text_w - margin
                        text_y = grid_max_y + margin
                    elif north_angle == 270:  # 上南下北：右上角是西南
                        text_x = grid_max_x + margin
                        text_y = grid_min_y - text_h - margin
                    elif north_angle == 0:  # 右北左南：左下角是西南
                        text_x = grid_min_x - text_w - margin
                        text_y = grid_max_y + margin
                    elif north_angle == 180:  # 左北右南：右上角是西南
                        text_x = grid_max_x + margin
                        text_y = grid_min_y - text_h - margin
                    else:
                        text_x = center_x - text_w / 2
                        text_y = center_y - text_h / 2
                else:
                    # 其他未处理的方位
                    text_x = center_x - text_w / 2
                    text_y = center_y - text_h / 2
