#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从网络扫描数据中提取router类型的设备数据
支持两种来源：
1. OS.Device Type 为 router 的数据
2. out.jsonl 中指纹识别为 RouterOS/Router 的 IP 对应数据
"""

import json
import os
from datetime import datetime
from collections import defaultdict

def load_router_ips_from_fingerprint(fingerprint_file):
    """
    从指纹识别结果文件中加载 RouterOS/Router 类型的 IP
    """
    router_ips = set()
    
    if not os.path.exists(fingerprint_file):
        print(f"警告: 指纹文件 {fingerprint_file} 不存在")
        return router_ips
    
    print(f"正在加载指纹识别结果: {fingerprint_file}")
    
    with open(fingerprint_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                ip = data.get('ip', '')
                fields = data.get('fields', {})
                os_info = fields.get('OS', {})
                device_type = os_info.get('Device Type', '')
                
                # 匹配 RouterOS 或包含 Router 的类型
                if 'Router' in device_type:
                    router_ips.add(ip)
            except:
                pass
    
    print(f"从指纹文件中找到 {len(router_ips)} 个 Router 类型 IP")
    return router_ips

def extract_routers(input_file, output_file, log_file, fingerprint_file=None):
    """
    从输入文件中提取Device Type为router的数据，以及指纹识别为Router的数据
    """
    
    # 加载指纹识别的 Router IP
    fingerprint_router_ips = set()
    if fingerprint_file:
        fingerprint_router_ips = load_router_ips_from_fingerprint(fingerprint_file)
    
    # 统计信息
    stats = {
        'total_records': 0,
        'router_count': 0,
        'from_os_field': 0,        # 来自 OS.Device Type 字段
        'from_fingerprint': 0,      # 来自指纹识别
        'os_distribution': defaultdict(int),
        'country_distribution': defaultdict(int),
        'asn_distribution': defaultdict(int),
        'services_distribution': defaultdict(int),
        'org_category_distribution': defaultdict(int),
        'errors': 0,
        'error_lines': []
    }
    
    router_lines = []
    found_ips = set()  # 避免重复
    
    print(f"开始处理文件: {input_file}")
    start_time = datetime.now()

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            original_line = line
            line = line.strip()
            if not line:
                continue
                
            stats['total_records'] += 1
            
            try:
                data = json.loads(line)
                
                # 遍历每个IP记录
                for ip, info in data.items():
                    if ip in found_ips:
                        continue
                    
                    is_router = False
                    source = None
                    
                    # 检查方式1: OS.Device Type 为 router
                    if isinstance(info, dict) and 'OS' in info:
                        os_info = info.get('OS', {})
                        if isinstance(os_info, dict):
                            device_type = os_info.get('Device Type', '')
                            if device_type == 'router':
                                is_router = True
                                source = 'os_field'
                    
                    # 检查方式2: IP 在指纹识别的 Router 列表中
                    if not is_router and ip in fingerprint_router_ips:
                        is_router = True
                        source = 'fingerprint'
                    
                    if is_router:
                        found_ips.add(ip)
                        stats['router_count'] += 1
                        
                        if source == 'os_field':
                            stats['from_os_field'] += 1
                        else:
                            stats['from_fingerprint'] += 1
                        
                        # 收集统计信息
                        if isinstance(info, dict):
                            os_info = info.get('OS', {})
                            if isinstance(os_info, dict):
                                os_name = os_info.get('OS', 'Unknown')
                                stats['os_distribution'][os_name] += 1
                            
                            country = info.get('Country', 'Unknown')
                            stats['country_distribution'][country] += 1
                            
                            asn = info.get('ASN', 'Unknown')
                            as_name = info.get('AS Name', 'Unknown')
                            stats['asn_distribution'][f"{asn} - {as_name}"] += 1
                            
                            services = info.get('Services', {})
                            if isinstance(services, dict):
                                for service in services.keys():
                                    stats['services_distribution'][service] += 1
                            
                            org = info.get('Org', {})
                            if isinstance(org, dict):
                                category = org.get('Category', 'Unknown')
                                stats['org_category_distribution'][category] += 1
                        
                        # 保存原始行数据
                        router_lines.append(original_line.rstrip('\n\r'))
                                
            except json.JSONDecodeError as e:
                stats['errors'] += 1
                if len(stats['error_lines']) < 10:
                    stats['error_lines'].append(f"Line {line_num}: {str(e)[:100]}")
            except Exception as e:
                stats['errors'] += 1
                if len(stats['error_lines']) < 10:
                    stats['error_lines'].append(f"Line {line_num}: {str(e)[:100]}")
            
            if stats['total_records'] % 100000 == 0:
                print(f"已处理 {stats['total_records']} 条记录, 发现 {stats['router_count']} 个router...")
    
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()
    
    # 写入router数据
    print(f"正在写入 {stats['router_count']} 条router数据到 {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in router_lines:
            f.write(line + '\n')

    # 写入日志
    print(f"正在写入日志到 {log_file}...")
    with open(log_file, 'w', encoding='utf-8-sig') as f:
        f.write("=" * 80 + "\n")
        f.write("Router数据提取日志\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"处理时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入文件: {input_file}\n")
        f.write(f"指纹文件: {fingerprint_file}\n")
        f.write(f"输出文件: {output_file}\n")
        f.write(f"处理耗时: {processing_time:.2f} 秒\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("基本统计\n")
        f.write("-" * 40 + "\n")
        f.write(f"总记录数: {stats['total_records']}\n")
        f.write(f"Router数量: {stats['router_count']}\n")
        f.write(f"  - 来自OS字段: {stats['from_os_field']}\n")
        f.write(f"  - 来自指纹识别: {stats['from_fingerprint']}\n")
        f.write(f"Router占比: {stats['router_count']/max(stats['total_records'],1)*100:.2f}%\n")
        f.write(f"解析错误数: {stats['errors']}\n\n")
        
        f.write("-" * 40 + "\n")
        f.write("操作系统分布 (Top 20)\n")
        f.write("-" * 40 + "\n")
        sorted_os = sorted(stats['os_distribution'].items(), key=lambda x: x[1], reverse=True)[:20]
        for os_name, count in sorted_os:
            f.write(f"  {os_name}: {count}\n")
        f.write("\n")
        
        f.write("-" * 40 + "\n")
        f.write("国家/地区分布 (Top 20)\n")
        f.write("-" * 40 + "\n")
        sorted_country = sorted(stats['country_distribution'].items(), key=lambda x: x[1], reverse=True)[:20]
        for country, count in sorted_country:
            f.write(f"  {country}: {count}\n")
        f.write("\n")
        
        f.write("-" * 40 + "\n")
        f.write("ASN分布 (Top 20)\n")
        f.write("-" * 40 + "\n")
        sorted_asn = sorted(stats['asn_distribution'].items(), key=lambda x: x[1], reverse=True)[:20]
        for asn, count in sorted_asn:
            f.write(f"  {asn}: {count}\n")
        f.write("\n")
        
        f.write("-" * 40 + "\n")
        f.write("服务分布\n")
        f.write("-" * 40 + "\n")
        sorted_services = sorted(stats['services_distribution'].items(), key=lambda x: x[1], reverse=True)
        for service, count in sorted_services:
            f.write(f"  {service}: {count}\n")
        f.write("\n")
        
        f.write("-" * 40 + "\n")
        f.write("组织类别分布\n")
        f.write("-" * 40 + "\n")
        sorted_org = sorted(stats['org_category_distribution'].items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_org:
            f.write(f"  {category}: {count}\n")
        f.write("\n")
        
        if stats['error_lines']:
            f.write("-" * 40 + "\n")
            f.write("错误信息 (前10条)\n")
            f.write("-" * 40 + "\n")
            for error in stats['error_lines']:
                f.write(f"  {error}\n")
    
    print("\n" + "=" * 50)
    print("处理完成!")
    print(f"总记录数: {stats['total_records']}")
    print(f"Router数量: {stats['router_count']}")
    print(f"  - 来自OS字段: {stats['from_os_field']}")
    print(f"  - 来自指纹识别: {stats['from_fingerprint']}")
    print(f"处理耗时: {processing_time:.2f} 秒")
    print("=" * 50)
    
    return stats

if __name__ == '__main__':
    input_file = 'Hassets_latest.json'
    fingerprint_file = 'out.jsonl'
    output_file = 'routers.json'
    log_file = 'log.txt'
    
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在!")
        exit(1)
    
    extract_routers(input_file, output_file, log_file, fingerprint_file)
