
// Copyright (c) Simon Fraser University & The Chinese University of Hong Kong. All rights reserved.
// Licensed under the MIT license.
//
// Dash Extendible Hashing
// Authors:
// Baotong Lu <btlu@cse.cuhk.edu.hk>
// Xiangpeng Hao <xiangpeng_hao@sfu.ca>
// Tianzheng Wang <tzwang@sfu.ca>

#pragma once

#include <immintrin.h>
#include <omp.h>

#include <bitset>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <shared_mutex>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "../util/hash.h"
#include "../util/pair.h"
#include "Hash.h"
#include "allocator.h"


//如果宏PMEM被定义的话，则编译一下代码
#ifdef PMEM
#include <libpmemobj.h>
#endif

uint64_t merge_time;

//命名空间extendible 可扩展哈希
namespace extendible {
//#define COUNTING 1
//#define PREALLOC 1

//键值对
//Value_t const char*
template <class T>
struct _Pair {
  T key;
  Value_t value;
};

//unit64_t  long 64位
//unit32_t  int 32位
//unit16_t  short int 16位
//unit8_t   char 8位

//mask都是为了做与运算得到需要的数据
//lockSet 为10000000 00000000 00000000 00000000
const uint32_t lockSet = ((uint32_t)1 << 31);
//lockMask 011111111 11111111 11111111 11111111
const uint32_t lockMask = ((uint32_t)1 << 31) - 1;
//00000000 00000000 00000000 00010000
const int overflowSet = 1 << 4;
//1111用来和bitmap做与运算 得到桶里面的数目
const int countMask = (1 << 4) - 1;
//00000000 11111111.....111111
const uint64_t tailMask = (1UL << 56) - 1;
//11111111000......000000
const uint64_t headerMask = ((1UL << 8) - 1) << 56;

//1111
const uint8_t overflowBitmapMask = (1 << 4) - 1;

constexpr size_t k_PairSize = 16;  // a k-v _Pair with a bit

//每一个Bucket有多少个键值对
constexpr size_t kNumPairPerBucket =
    14; /* it is determined by the usage of the fingerprint*/

//指纹的大小为1B
constexpr size_t kFingerBits = 8;

//指纹初始化为 11111111
constexpr size_t kMask = (1 << kFingerBits) - 1;
//每个段多少个正常的桶
const constexpr size_t kNumBucket =
    64; /* the number of normal buckets in one segment*/
//每个段两个stash
constexpr size_t stashBucket =
    2; /* the number of stash buckets in one segment*/

//allocbitmap 初始化00000000 00000000 00111111 11111111
constexpr int allocMask = (1 << kNumPairPerBucket) - 1;

//判断在哪个正常桶中 bucket 需要6位 因为只有64个桶 111111
constexpr size_t bucketMask = ((1 << (int)log2(kNumBucket)) - 1);

//判断在那个stash里 1位即可 因为只有两个桶 1
constexpr size_t stashMask = (1 << (int)log2(stashBucket)) - 1;
constexpr uint8_t stashHighMask = ~((uint8_t)stashMask);

#define BUCKET_INDEX(hash) ((hash >> kFingerBits) & bucketMask)
#define GET_COUNT(var) ((var)&countMask)
#define GET_MEMBER(var) (((var) >> 4) & allocMask)
//对member取反
#define GET_INVERSE_MEMBER(var) ((~((var) >> 4)) & allocMask)
#define GET_BITMAP(var) ((var) >> 18)

//比较str1和str2内存区间的前len1个字符是不是一样
inline bool var_compare(char *str1, char *str2, int len1, int len2) {
  if (len1 != len2) return false;
  return !memcmp(str1, str2, len1);
}

template <class T>
struct Bucket {
  inline int find_empty_slot() {
    //做与运算得到键值对的数目
    if (GET_COUNT(bitmap) == kNumPairPerBucket) {
      return -1;
    }
    //对bitmap取反，得到空余位置
    auto mask = ~(GET_BITMAP(bitmap));
    //返回右起第一个1之后的0的个数
    return __builtin_ctz(mask);
  }

  /*true indicates overflow, needs extra check in the stash*/
  //返回存在溢出桶的个数
  inline bool test_overflow() { return overflowCount; }

  //overflowbitmap =overflowbit（0000）+overflowbitmap（0000）组成每个四位
  //test_stash_check 用来判断有没有溢出
  inline bool test_stash_check() { return (overflowBitmap & overflowSet); }


  //overflowSet 00000000 00000000 00000000 00010000
  //将overflowbit置为0
  inline void clear_stash_check() {
    overflowBitmap = overflowBitmap & (~overflowSet);
  }

  /*meta_hash 指纹
   *neighbor 邻居桶
   */
  //当你往stash桶中插入数据的时候，需要修改正常桶的有关于overflowbucket的元数据
  inline void set_indicator(uint8_t meta_hash, Bucket<T> *neighbor,
                            uint8_t pos) {
    //得到overflowBitmap
    int mask = overflowBitmap & overflowBitmapMask;
    //取反
    mask = ~mask;
    //返回右起第一个1之后的0的个数
    //比如现在overflowBitmap为1110 说明下一个可以用的位置是第2个1 index就是1
    auto index = __builtin_ctz(mask);

    //当index等于4的时候，说明这个桶存不了溢出字段的指纹了
    if (index < 4) {
      //存储溢出的指纹
      finger_array[14 + index] = meta_hash;
      //将用过的index置为1
      overflowBitmap = ((uint8_t)(1 << index) | overflowBitmap);
      //记录这个fingerprint存储在哪个stash中
      overflowIndex =
          (overflowIndex & (~(3 << (index * 2)))) | (pos << (index * 2));
    } else {
      /*
       * 这一部分说明target桶的4个存储溢出指纹的指标用完了，看看可不可以存储到邻居桶中
       * 对邻居桶的元数据进行一样的修改
       */
      mask = neighbor->overflowBitmap & overflowBitmapMask;
      mask = ~mask;
      index = __builtin_ctz(mask);
      if (index < 4) {
        neighbor->finger_array[14 + index] = meta_hash;
        neighbor->overflowBitmap =
            ((uint8_t)(1 << index) | neighbor->overflowBitmap);
        neighbor->overflowMember =
            ((uint8_t)(1 << index) | neighbor->overflowMember);
        neighbor->overflowIndex =
            (neighbor->overflowIndex & (~(3 << (index * 2)))) |
            (pos << (index * 2));
      } else { /*overflow, increase count 这时候需要遍历两个stash桶*/
        overflowCount++;
      }
    }
    //将overflowbit置为1
    overflowBitmap = overflowBitmap | overflowSet;
  }

  /*both clear this bucket and its neighbor bucket*/
  //删出某个key，假设这个key存储在stash中，那么需要改变overflow相关的元数据
  inline void unset_indicator(uint8_t meta_hash, Bucket<T> *neighbor, T key,
                              uint64_t pos) {
    /*also needs to ensure that this meta_hash must belongs to other bucket*/
    bool clear_success = false;
    int mask1 = overflowBitmap & overflowBitmapMask;
    //查看这个key在不在探测桶里面
    for (int i = 0; i < 4; ++i) {
      //比较bitmap对应的位置是否为1，再比较指纹是否一致，再看看是不是属于探测桶，再看看是不是都在同一个stash桶
      if (CHECK_BIT(mask1, i) && (finger_array[14 + i] == meta_hash) &&
          (((1 << i) & overflowMember) == 0) &&
          (((overflowIndex >> (2 * i)) & stashMask) == pos)) {
        //重置所有的元数据
        overflowBitmap = overflowBitmap & ((uint8_t)(~(1 << i)));
        overflowIndex = overflowIndex & (~(3 << (i * 2)));
        assert(((overflowIndex >> (i * 2)) & stashMask) == 0);
        clear_success = true;
        break;
      }
    }

    //不在target桶中，看看是不是在探测桶中
    int mask2 = neighbor->overflowBitmap & overflowBitmapMask;
    if (!clear_success) {
      for (int i = 0; i < 4; ++i) {
        if (CHECK_BIT(mask2, i) &&
            (neighbor->finger_array[14 + i] == meta_hash) &&
            (((1 << i) & neighbor->overflowMember) != 0) &&
            (((neighbor->overflowIndex >> (2 * i)) & stashMask) == pos)) {
          neighbor->overflowBitmap =
              neighbor->overflowBitmap & ((uint8_t)(~(1 << i)));
          neighbor->overflowMember =
              neighbor->overflowMember & ((uint8_t)(~(1 << i)));
          neighbor->overflowIndex = neighbor->overflowIndex & (~(3 << (i * 2)));
          assert(((neighbor->overflowIndex >> (i * 2)) & stashMask) == 0);
          clear_success = true;
          break;
        }
      }
    }
    //如果没有存储指纹的话，只需要将overflowcount--即可
    if (!clear_success) {
      overflowCount--;
    }
    //得到二者的bitmap
    mask1 = overflowBitmap & overflowBitmapMask;
    mask2 = neighbor->overflowBitmap & overflowBitmapMask;

    //target桶中的溢出指纹来自它左边的桶
    //overflowcount为0
    //target桶中溢出指纹到它的探测桶了
    //修改overflowbit 置为0
    if (((mask1 & (~overflowMember)) == 0) && (overflowCount == 0) &&
        ((mask2 & neighbor->overflowMember) == 0)) {
      clear_stash_check();
    }
  }

  /*
   * 唯一性检查函数
   * meta_hash 指纹
   * key 查找的key
   * neighbor 邻居桶
   * stash 存储桶
   */
  int unique_check(uint8_t meta_hash, T key, Bucket<T> *neighbor,
                   Bucket<T> *stash) {
    //首先看target桶和probe桶中有没有数据
    if ((check_and_get(meta_hash, key, false) != NONE) ||
        (neighbor->check_and_get(meta_hash, key, true) != NONE)) {
      return -1;
    }
    //如果溢出位不为0
    //test_overflow()是看count为不为0，count不为0的话需要对stash直接进行遍历
    //如过count为0的话，说明我们可以借助存储的4个overflow fingerprint来进行查找数据

    if (test_stash_check()) {
      auto test_stash = false;
      if (test_overflow()) {
        //count为1的话 需要探测stash桶
        test_stash = true;
      } else {
        //如果count为0的话，可以借助一下overflowfingerprint来进行查找，查找邻居桶和探测桶
        //如果有一样的指纹的话，那么就需要探测stash桶
        int mask = overflowBitmap & overflowBitmapMask;
        if (mask != 0) {
          for (int i = 0; i < 4; ++i) {
            if (CHECK_BIT(mask, i) && (finger_array[14 + i] == meta_hash) &&
                (((1 << i) & overflowMember) == 0)) {
              test_stash = true;
              goto STASH_CHECK;
            }
          }
        }
        mask = neighbor->overflowBitmap & overflowBitmapMask;
        if (mask != 0) {
          for (int i = 0; i < 4; ++i) {
            if (CHECK_BIT(mask, i) &&
                (neighbor->finger_array[14 + i] == meta_hash) &&
                (((1 << i) & neighbor->overflowMember) != 0)) {
              test_stash = true;
              break;
            }
          }
        }
      }
    STASH_CHECK:
      //遍历所有的stash桶 看看找不找的到
      if (test_stash == true) {
        for (int i = 0; i < stashBucket; ++i) {
          Bucket *curr_bucket = stash + i;
          if (curr_bucket->check_and_get(meta_hash, key, false) != NONE) {
            return -1;
          }
        }
      }
    }
    return 0;
  }

  //bitmap=alloc+member+counter
  //得到本身就属于当前桶的数据，而不是从其它桶迁移过来
  inline int get_current_mask() {
    int mask = GET_BITMAP(bitmap) & GET_INVERSE_MEMBER(bitmap);
    return mask;
  }

  /*
   * 唯一性检查，查找到的话 返回value
   */
  Value_t check_and_get(uint8_t meta_hash, T key, bool probe) {
    int mask = 0;
    //先查找指纹是不是一致的
    SSE_CMP8(finger_array, meta_hash);
    //判断查找的桶是不是探测桶，如果不是探测桶的话 只需要查找自己的成员 是探测桶的话 那么只查找member为1的就行
    if (!probe) {
      mask = mask & GET_BITMAP(bitmap) & (~GET_MEMBER(bitmap));
    } else {
      mask = mask & GET_BITMAP(bitmap) & GET_MEMBER(bitmap);
    }

    if (mask == 0) {
      return NONE;
    }
    /*
     * 作者采用指针存储变长键的类型
     * 指针的话需要一个一个的去比较key
     */
    if constexpr (std::is_pointer_v<T>) {
      /* variable-length key*/
      string_key *_key = reinterpret_cast<string_key *>(key);
      for (int i = 0; i < 14; i += 1) {
        if (CHECK_BIT(mask, i) &&
            (var_compare((reinterpret_cast<string_key *>(_[i].key))->key,
                         _key->key,
                         (reinterpret_cast<string_key *>(_[i].key))->length,
                         _key->length))) {
          return _[i].value;
        }
      }
    } else {
      /*fixed-length key*/
      /*
       * 循环展开
       */
      /*loop unrolling*/
      for (int i = 0; i < 12; i += 4) {
        if (CHECK_BIT(mask, i) && (_[i].key == key)) {
          return _[i].value;
        }

        if (CHECK_BIT(mask, i + 1) && (_[i + 1].key == key)) {
          return _[i + 1].value;
        }

        if (CHECK_BIT(mask, i + 2) && (_[i + 2].key == key)) {
          return _[i + 2].value;
        }

        if (CHECK_BIT(mask, i + 3) && (_[i + 3].key == key)) {
          return _[i + 3].value;
        }
      }

      if (CHECK_BIT(mask, 12) && (_[12].key == key)) {
        return _[12].value;
      }

      if (CHECK_BIT(mask, 13) && (_[13].key == key)) {
        return _[13].value;
      }
    }
    return NONE;
  }

  //修改bitmap的 alloc member counter位
  inline void set_hash(int index, uint8_t meta_hash, bool probe) {
    finger_array[index] = meta_hash;
    uint32_t new_bitmap = bitmap | (1 << (index + 18));
    //如果是探测桶的话 还需要修改memmber位
    if (probe) {
      new_bitmap = new_bitmap | (1 << (index + 4));
    }
    //counter++即可
    new_bitmap += 1;
    bitmap = new_bitmap;
  }

  //得到某个index的指纹，设置非stash的那一块的元数据
  inline uint8_t get_hash(int index) { return finger_array[index]; }

  //删除掉index 对应的member alloc都需要变为0，同时count--
  inline void unset_hash(int index, bool nt_flush = false) {
    uint32_t new_bitmap =
        bitmap & (~(1 << (index + 18))) & (~(1 << (index + 4)));
    //count小于等于kNum且大于0
    assert(GET_COUNT(bitmap) <= kNumPairPerBucket);
    assert(GET_COUNT(bitmap) > 0);
    new_bitmap -= 1;
#ifdef PMEM
    if (nt_flush) {
      Allocator::NTWrite32(reinterpret_cast<uint32_t *>(&bitmap), new_bitmap);
    } else {
      bitmap = new_bitmap;
    }
#else
    bitmap = new_bitmap;
#endif
  }

  //对桶加锁
  inline void get_lock() {
    uint32_t new_value = 0;
    uint32_t old_value = 0;
    do {
      while (true) {
        //原子读取version_lock gcc
        //ACQUIRE 本线程中，所有后续的读操作必须在本条原子操作完成后执行
        old_value = __atomic_load_n(&version_lock, __ATOMIC_ACQUIRE);
        //如果现在没锁的话 指的是old_value 第一位为0
        if (!(old_value & lockSet)) {
          //oldvalue后31位不变 第一位依然为0
          old_value &= lockMask;
          break;
        }
      }
      //将第一位变为1 上锁
      new_value = old_value | lockSet;
    } while (!CAS(&version_lock, &old_value, new_value));
  }

  //得到锁
  inline bool try_get_lock() {
    //读取32位的锁
    uint32_t v = __atomic_load_n(&version_lock, __ATOMIC_ACQUIRE);
    //如果现在锁被加上了，返回错误，没得到
    if (v & lockSet) {
      return false;
    }
    //旧值等于v
    auto old_value = v & lockMask;
    //新值等于加上锁，将第一位变成1
    auto new_value = v | lockSet;

    return CAS(&version_lock, &old_value, new_value);
  }

  //释放锁
  inline void release_lock() {
    uint32_t v = version_lock;
    //本线程中，所有之前的写操作完成后才能执行本条原子操作
    //版本号+1然后，将第一位变为0
    __atomic_store_n(&version_lock, v + 1 - lockSet, __ATOMIC_RELEASE);
  }

  //查看是否加上锁了
  /*if the lock is set, return true*/
  inline bool test_lock_set(uint32_t &version) {
    version = __atomic_load_n(&version_lock, __ATOMIC_ACQUIRE);
    return (version & lockSet) != 0;
  }

  //查看版本号是否变了
  // test whether the version has change, if change, return true
  inline bool test_lock_version_change(uint32_t old_version) {
    auto value = __atomic_load_n(&version_lock, __ATOMIC_ACQUIRE);
    return (old_version != value);
  }

  /*
   * 插入操作
   * 1.找到空闲桶
   * 2.插入数据，并且持久化
   * 3.修改元数据
   * question:为什么不持久化指纹
   */
  int Insert(T key, Value_t value, uint8_t meta_hash, bool probe) {
    //得到空闲槽的索引
    auto slot = find_empty_slot();
    assert(slot < kNumPairPerBucket);
    //如果slot=-1 返回-1
    if (slot == -1) {
      return -1;
    }
    //将key和value写入到slot中，一个slot对应一个键值对
    _[slot].value = value;
    _[slot].key = key;
#ifdef PMEM
    //传入地址和大小即可，持久化数据
    Allocator::Persist(&_[slot], sizeof(_[slot]));
#endif
    //设置元数据
    set_hash(slot, meta_hash, probe);
    return 0;
  }

  /*if delete success, then return 0, else return -1*/
  int Delete(T key, uint8_t meta_hash, bool probe) {
    /*do the simd and check the key, then do the delete operation*/
    int mask = 0;
    SSE_CMP8(finger_array, meta_hash);
    if (!probe) {
      mask = mask & GET_BITMAP(bitmap) & (~GET_MEMBER(bitmap));
    } else {
      mask = mask & GET_BITMAP(bitmap) & GET_MEMBER(bitmap);
    }

    /*loop unrolling*/
    if constexpr (std::is_pointer_v<T>) {
      string_key *_key = reinterpret_cast<string_key *>(key);
      /*loop unrolling*/
      if (mask != 0) {
        for (int i = 0; i < 12; i += 4) {
          if (CHECK_BIT(mask, i) &&
              (var_compare((reinterpret_cast<string_key *>(_[i].key))->key,
                           _key->key,
                           (reinterpret_cast<string_key *>(_[i].key))->length,
                           _key->length))) {
            unset_hash(i, false);
            return 0;
          }

          if (CHECK_BIT(mask, i + 1) &&
              (var_compare(
                  reinterpret_cast<string_key *>(_[i + 1].key)->key, _key->key,
                  (reinterpret_cast<string_key *>(_[i + 1].key))->length,
                  _key->length))) {
            unset_hash(i + 1, false);
            return 0;
          }

          if (CHECK_BIT(mask, i + 2) &&
              (var_compare(
                  reinterpret_cast<string_key *>(_[i + 2].key)->key, _key->key,
                  (reinterpret_cast<string_key *>(_[i + 2].key))->length,
                  _key->length))) {
            unset_hash(i + 2, false);
            return 0;
          }

          if (CHECK_BIT(mask, i + 3) &&
              (var_compare(
                  reinterpret_cast<string_key *>(_[i + 3].key)->key, _key->key,
                  (reinterpret_cast<string_key *>(_[i + 3].key))->length,
                  _key->length))) {
            unset_hash(i + 3, false);
            return 0;
          }
        }

        if (CHECK_BIT(mask, 12) &&
            (var_compare(reinterpret_cast<string_key *>(_[12].key)->key,
                         _key->key,
                         (reinterpret_cast<string_key *>(_[12].key))->length,
                         _key->length))) {
          unset_hash(12, false);
          return 0;
        }

        if (CHECK_BIT(mask, 13) &&
            (var_compare(reinterpret_cast<string_key *>(_[13].key)->key,
                         _key->key,
                         (reinterpret_cast<string_key *>(_[13].key))->length,
                         _key->length))) {
          unset_hash(13, false);
          return 0;
        }
      }

    } else {
      if (mask != 0) {
        for (int i = 0; i < 12; i += 4) {
          if (CHECK_BIT(mask, i) && (_[i].key == key)) {
            unset_hash(i, false);
            return 0;
          }

          if (CHECK_BIT(mask, i + 1) && (_[i + 1].key == key)) {
            unset_hash(i + 1, false);
            return 0;
          }

          if (CHECK_BIT(mask, i + 2) && (_[i + 2].key == key)) {
            unset_hash(i + 2, false);
            return 0;
          }

          if (CHECK_BIT(mask, i + 3) && (_[i + 3].key == key)) {
            unset_hash(i + 3, false);
            return 0;
          }
        }

        if (CHECK_BIT(mask, 12) && (_[12].key == key)) {
          unset_hash(12, false);
          return 0;
        }

        if (CHECK_BIT(mask, 13) && (_[13].key == key)) {
          unset_hash(13, false);
          return 0;
        }
      }
    }
    return -1;
  }


  int Insert_with_noflush(T key, Value_t value, uint8_t meta_hash, bool probe) {
    auto slot = find_empty_slot();
    /* this branch can be removed*/
    assert(slot < kNumPairPerBucket);
    if (slot == -1) {
      std::cout << "Cannot find the empty slot, for key " << key << std::endl;
      return -1;
    }
    _[slot].value = value;
    _[slot].key = key;
    set_hash(slot, meta_hash, probe);
    return 0;
  }

  //displace 之后存在一个空位置，往这个slot里面插入数据
  void Insert_displace(T key, Value_t value, uint8_t meta_hash, int slot,
                       bool probe) {
    _[slot].value = value;
    _[slot].key = key;
#ifdef PMEM
    Allocator::Persist(&_[slot], sizeof(_Pair<T>));
#endif
    set_hash(slot, meta_hash, probe);
  }

  void Insert_displace_with_noflush(T key, Value_t value, uint8_t meta_hash,
                                    int slot, bool probe) {
    _[slot].value = value;
    _[slot].key = key;
    set_hash(slot, meta_hash, probe);
  }

  /* Find the displacment element in this bucket*/
  /*找到该桶中membership为1的slot，从而当作Displace的项*/
  //从b中找到slot displace到b-1
  inline int Find_org_displacement() {
    uint32_t mask = GET_INVERSE_MEMBER(bitmap);
    if (mask == 0) {
      return -1;
    }
    return __builtin_ctz(mask);
  }

  /*find element that it is in the probe*/
  //从b+1中找到位置从而移动到b+2
  inline int Find_probe_displacement() {
    uint32_t mask = GET_MEMBER(bitmap);
    if (mask == 0) {
      return -1;
    }
    return __builtin_ctz(mask);
  }

  //重置锁 0
  inline void resetLock() { version_lock = 0; }

  //重置overflow有关的标志位
  inline void resetOverflowFP() {
    overflowBitmap = 0;
    overflowIndex = 0;
    overflowMember = 0;
    overflowCount = 0;
    clear_stash_check();
  }

  uint32_t version_lock;     //桶级锁
  uint32_t bitmap;          // allocation bitmap + pointer bitmap + counter
  uint8_t finger_array[18]; /*only use the first 14 bytes, can be accelerated by
                               SSE instruction,0-13 for finger, 14-17 for
                               overflowed*/
  uint8_t overflowBitmap; //overflowbit(0000或者0001，用来标识这个桶有没有溢出指纹都)+4个溢出指纹的bitmap
  uint8_t overflowIndex;//4个指纹对应的位置，每个用2位表示，但好像只有两个stash桶 可能只要1位就行
  uint8_t overflowMember; /*overflowmember indicates membership of the overflow
                             fingerprint*/
  uint8_t overflowCount;
  uint8_t unused[2];

  _Pair<T> _[kNumPairPerBucket];
};

template <class T>
struct Table;


template <class T>
struct Directory {
  typedef Table<T> *table_p; //table的初始地址
  uint32_t global_depth; //全局深度
  uint32_t version; //目录的版本号，用来并发控制,以及锁位
  uint32_t depth_count;//
  table_p _[0];  //_ 一个数组 其中的内容是指向Table的指针

  /*
   * capacity:容量大小，指的是目录有多少个
   */
  Directory(size_t capacity, size_t _version) {
    version = _version;
    global_depth = static_cast<size_t>(log2(capacity));
    depth_count = 0;
  }
  /*动态分配一块内存
   * void* 抽象的指针
   * PMEMoid 持久指针，用于访问持久性内存，本质上是一个结构体
   * PMEMobjpool 指向内存池的指针
   */

  static void New(PMEMoid *dir, size_t capacity, size_t version) {
#ifdef PMEM
    auto callback = [](PMEMobjpool *pool, void *ptr, void *arg) {
      //arg指向一个tuple元组，第一个size_t为capacity 第二个size_t为version
      auto value_ptr = reinterpret_cast<std::tuple<size_t, size_t> *>(arg);
      //目录地址，指向一个目录
      auto dir_ptr = reinterpret_cast<Directory *>(ptr);
      dir_ptr->version = std::get<1>(*value_ptr);
      dir_ptr->global_depth =
          static_cast<size_t>(log2(std::get<0>(*value_ptr)));

      size_t cap = std::get<0>(*value_ptr);
      //持久化一块内存，pool为内存池，起始地址为dir_ptr，大小就是目录元数据的大小+64*容量（00 01 10 11,假设目录是这样子的话，那容量为4）
      pmemobj_persist(pool, dir_ptr,
                      sizeof(Directory<T>) + sizeof(uint64_t) * cap);
      return 0;
    };
    std::tuple callback_args = {capacity, version};
    // pmemobj_alloc(instance_->pm_pool_, pm_ptr, size,TOID_TYPE_NUM(char), alloc_constr, arg)
    Allocator::Allocate(dir, kCacheLineSize,
                        sizeof(Directory<T>) + sizeof(table_p) * capacity,
                        callback, reinterpret_cast<void *>(&callback_args));
#else
    Allocator::Allocate((void **)dir, kCacheLineSize, sizeof(Directory<T>));
    new (*dir) Directory(capacity, version, tables);
#endif
  }
};

/*thread local table allcoation pool*/
//PM的动态内存分配
template <class T>
struct TlsTablePool {
  //相当于segment
  static Table<T> *all_tables;
  //一个结构体对象PMEoid是一个结构体对象
  static PMEMoid p_all_tables;
  //8B的原子对象
  static std::atomic<uint32_t> all_allocated;
  //
  static const uint32_t kAllTables = 327680;

  static void AllocateMore() {
    auto callback = [](PMEMobjpool *pool, void *ptr, void *arg) { return 0; };
    std::pair callback_para(0, nullptr);
    Allocator::Allocate(&p_all_tables, kCacheLineSize,
                        sizeof(Table<T>) * kAllTables, callback,
                        reinterpret_cast<void *>(&callback_para));
    //new出来的持久性内存指针强转为all_tables
    all_tables = reinterpret_cast<Table<T> *>(pmemobj_direct(p_all_tables));
    //void *memset(void *str, int c, size_t n) str:要被填充内存块 c：要填充的值  len:填充大小
    memset((void *)all_tables, 0, sizeof(Table<T>) * kAllTables);
    all_allocated = 0;
    printf("MORE ");
  }

  TlsTablePool() {}
  static void Initialize() { AllocateMore(); }

  Table<T> *tables = nullptr;
  static const uint32_t kTables = 128;
  uint32_t allocated = kTables;

  void TlsPrepare() {
  retry:
    uint32_t n = all_allocated.fetch_add(kTables);
    if (n == kAllTables) {
      AllocateMore();
      abort();
      goto retry;
    }
    tables = all_tables + n;
    allocated = 0;
  }

  Table<T> *Get() {
    if (allocated == kTables) {
      TlsPrepare();
    }
    return &tables[allocated++];
  }
};

template <class T>
std::atomic<uint32_t> TlsTablePool<T>::all_allocated(0);
template <class T>
Table<T> *TlsTablePool<T>::all_tables = nullptr;
template <class T>
PMEMoid TlsTablePool<T>::p_all_tables = OID_NULL;

/* the segment class*/
template <class T>
struct Table {
  //分配内存
  static void New(PMEMoid *tbl, size_t depth, PMEMoid pp) {
#ifdef PMEM
#ifdef PREALLOC
    thread_local TlsTablePool<T> tls_pool;
    auto ptr = tls_pool.Get();
    ptr->local_depth = depth;
    ptr->next = pp;
    *tbl = pmemobj_oid(ptr);
#else
      auto callback = [](PMEMobjpool *pool, void *ptr, void *arg) {
      auto value_ptr = reinterpret_cast<std::pair<size_t, PMEMoid> *>(arg);
      auto table_ptr = reinterpret_cast<Table<T> *>(ptr);
      table_ptr->local_depth = value_ptr->first;
      table_ptr->next = value_ptr->second;
      table_ptr->state = -3; /*NEW*/
      memset(&table_ptr->lock_bit, 0, sizeof(PMEMmutex) * 2);

      //一个段的桶由kNumBucket和stashBucket组成，为每个通过赋值
      int sumBucket = kNumBucket + stashBucket;
      for (int i = 0; i < sumBucket; ++i) {
        auto curr_bucket = table_ptr->bucket + i;
        //一个int是4B 分64个4B，相当于一个桶256B
        memset(curr_bucket, 0, 64);
      }
      //将在DRAM中的内存持久化到PM上
      pmemobj_persist(pool, table_ptr, sizeof(Table<T>));
      return 0;
    };
    std::pair callback_para(depth, pp);
    Allocator::Allocate(tbl, kCacheLineSize, sizeof(Table<T>), callback,
                        reinterpret_cast<void *>(&callback_para));
#endif
#else
    Allocator::ZAllocate((void **)tbl, kCacheLineSize, sizeof(Table<T>));
    (*tbl)->local_depth = depth;
    (*tbl)->next = pp;
#endif
  };
  ~Table(void) {}

  //
  bool Acquire_and_verify(size_t _pattern) {
    bucket->get_lock();
    if (pattern != _pattern) {
      bucket->release_lock();
      return false;
    } else {
      return true;
    }
  }
  //将这个bucket之后的所有bucket都加锁
  void Acquire_remaining_locks() {
    for (int i = 1; i < kNumBucket; ++i) {
      auto curr_bucket = bucket + i;
      curr_bucket->get_lock();
    }
  }

  //将这个bucket之后的bucket的锁都释放
  void Release_all_locks() {
    for (int i = 0; i < kNumBucket; ++i) {
      auto curr_bucket = bucket + i;
      curr_bucket->release_lock();
    }
  }

  int Insert(T key, Value_t value, size_t key_hash, uint8_t meta_hash,
             Directory<T> **);
  //分割时候的插入
  void Insert4split(T key, Value_t value, size_t key_hash, uint8_t meta_hash);
  //有唯一性检查
  void Insert4splitWithCheck(T key, Value_t value, size_t key_hash,
                             uint8_t meta_hash); /*with uniqueness check*/
  //merge操作
  void Insert4merge(T key, Value_t value, size_t key_hash, uint8_t meta_hash,
                    bool flag = false);
  //分割函数，返回一个指针，指针指向新段
  Table<T> *Split(size_t);
  void HelpSplit(Table<T> *);
  void Merge(Table<T> *, bool flag = false);
  int Delete(T key, size_t key_hash, uint8_t meta_hash, Directory<T> **_dir);

  //将b+1的数据移动到b+2中，移动成功的话需要修改的数据
  //b+1的bitmap和membership
  //b+2的bitmap和membership，相当于探测桶的插入
  int Next_displace(Bucket<T> *target, Bucket<T> *neighbor,
                    Bucket<T> *next_neighbor, T key, Value_t value,
                    uint8_t meta_hash) {
    //查找b+1中本身的数据
    int displace_index = neighbor->Find_org_displacement();
    //看看b+2还有没有空余的位置，没有的话返回-1 向后displace失败
    if ((GET_COUNT(next_neighbor->bitmap) != kNumPairPerBucket) &&
        (displace_index != -1)) {
      //下一个bucket插入数据，相当于探测桶
      next_neighbor->Insert(neighbor->_[displace_index].key,
                            neighbor->_[displace_index].value,
                            neighbor->finger_array[displace_index], true);
      next_neighbor->release_lock();
#ifdef PMEM
      //持久化bitmap
      Allocator::Persist(&next_neighbor->bitmap, sizeof(next_neighbor->bitmap));
#endif
      //插入数据,这里可能会导致有重复数据
      neighbor->unset_hash(displace_index);
      neighbor->Insert_displace(key, value, meta_hash, displace_index, true);
      neighbor->release_lock();
#ifdef PMEM
      Allocator::Persist(&neighbor->bitmap, sizeof(neighbor->bitmap));
#endif
      target->release_lock();
#ifdef COUNTING
      __sync_fetch_and_add(&number, 1);
#endif
      return 0;
    }
    return -1;
  }
  //将b中的b-1的数据移动到b-1
  //修改b-1的bitmap和membership
  //修改b的bitmap和membership
  int Prev_displace(Bucket<T> *target, Bucket<T> *prev_neighbor,
                    Bucket<T> *neighbor, T key, Value_t value,
                    uint8_t meta_hash) {
    int displace_index = target->Find_probe_displacement();
    if ((GET_COUNT(prev_neighbor->bitmap) != kNumPairPerBucket) &&
        (displace_index != -1)) {
      prev_neighbor->Insert(target->_[displace_index].key,
                            target->_[displace_index].value,
                            target->finger_array[displace_index], false);
      prev_neighbor->release_lock();
#ifdef PMEM
      Allocator::Persist(&prev_neighbor->bitmap, sizeof(prev_neighbor->bitmap));
#endif
      target->unset_hash(displace_index);
      target->Insert_displace(key, value, meta_hash, displace_index, false);
      target->release_lock();
#ifdef PMEM
      Allocator::Persist(&target->bitmap, sizeof(target->bitmap));
#endif
      neighbor->release_lock();
#ifdef COUNTING
      __sync_fetch_and_add(&number, 1);
#endif
      return 0;
    }
    return -1;
  }

  int Stash_insert(Bucket<T> *target, Bucket<T> *neighbor, T key, Value_t value,
                   uint8_t meta_hash, int stash_pos) {
    for (int i = 0; i < stashBucket; ++i) {
      Bucket<T> *curr_bucket =
          bucket + kNumBucket + ((stash_pos + i) & stashMask);
      if (GET_COUNT(curr_bucket->bitmap) < kNumPairPerBucket) {
        curr_bucket->Insert(key, value, meta_hash, false);
#ifdef PMEM
        Allocator::Persist(&curr_bucket->bitmap, sizeof(curr_bucket->bitmap));
#endif
        //设置overflow相关的元数据
        target->set_indicator(meta_hash, neighbor, (stash_pos + i) & stashMask);
#ifdef COUNTING
        __sync_fetch_and_add(&number, 1);
#endif
        return 0;
      }
    }
    return -1;
  }

  void recoverMetadata() {
    Bucket<T> *curr_bucket, *neighbor_bucket;
    /*reset the lock and overflow meta-data*/
    uint64_t knumber = 0;
    //这一部分主要是删除重复的数据
    for (int i = 0; i < kNumBucket; ++i) {
      curr_bucket = bucket + i;
      //释放桶级锁
      curr_bucket->resetLock();
      //删除overflow相关的元数据
      curr_bucket->resetOverflowFP();
      //得到neighbor的桶
      neighbor_bucket = bucket + ((i + 1) & bucketMask);
      for (int j = 0; j < kNumPairPerBucket; ++j) {
       //mask 得到本身就属于当前的桶的数据。而不是从其它桶迁移过来的
       //如果neighbor有重复的数据的话，删除
        int mask = curr_bucket->get_current_mask();
        if (CHECK_BIT(mask, j) && (neighbor_bucket->check_and_get(
                                       curr_bucket->finger_array[j],
                                       curr_bucket->_[j].key, true) != NONE)) {
          //找到的话，删除即可
          curr_bucket->unset_hash(j);
        }
      }

#ifdef COUNTING
      knumber += __builtin_popcount(GET_BITMAP(curr_bucket->bitmap));
#endif
    }

    /*scan the stash buckets and re-insert the overflow FP to initial buckets*/
    //遍历stash中的数据 恢复元数据
    for (int i = 0; i < stashBucket; ++i) {
      curr_bucket = bucket + kNumBucket + i;
      //将锁释放
      curr_bucket->resetLock();
#ifdef COUNTING
      knumber += __builtin_popcount(GET_BITMAP(curr_bucket->bitmap));
#endif
      uint64_t key_hash;
      auto mask = GET_BITMAP(curr_bucket->bitmap);
      for (int j = 0; j < kNumPairPerBucket; ++j) {
        if (CHECK_BIT(mask, j)) {
          //看看是key是不是指针
          if constexpr (std::is_pointer_v<T>) {
            auto curr_key = curr_bucket->_[j].key;
            key_hash = h(curr_key->key, curr_key->length);
          } else {
            key_hash = h(&(curr_bucket->_[j].key), sizeof(Key_t));
          }
          /*compute the initial bucket*/
          //计算数据本应该在的桶
          auto bucket_ix = BUCKET_INDEX(key_hash);
          //拿出指纹
          auto meta_hash = ((uint8_t)(key_hash & kMask));  // the last 8 bits
          auto org_bucket = bucket + bucket_ix;
          //计算邻居桶
          auto neighbor_bucket = bucket + ((bucket_ix + 1) & bucketMask);
          //恢复元数据
          org_bucket->set_indicator(meta_hash, neighbor_bucket, i);
        }
      }
    }
#ifdef COUNTING
    number = knumber;
#endif
    /* No need to flush these meta-data because persistent or not does not
     * influence the correctness*/
  }

  char dummy[48];
  Bucket<T> bucket[kNumBucket + stashBucket];
  size_t local_depth; //段的本地深度
  size_t pattern;     //当前段的标号 比如00 01 10 11
  int number;         //版本号
  PMEMoid next;    //指向下一个段
  int state; /*-1 means this bucket is merging, -2 means this bucket is
                splitting (SPLITTING), 0 meanning normal bucket, -3 means new
                bucket (NEW)*/
  PMEMmutex
      lock_bit; /* for the synchronization of the lazy recovery in one segment*/
};



/* it needs to verify whether this bucket has been deleted...*/
//question:为什么没有持久化指纹
template <class T>
int Table<T>::Insert(T key, Value_t value, size_t key_hash, uint8_t meta_hash,
                     Directory<T> **_dir) {
RETRY:
  /*we need to first do the locking and then do the verify*/
  //LSB作为桶的索引，但向右移动的8位再计算 取右bucketMask 说明连续的.....00000000-.....11111111都在一个桶里？用了局部性？
  auto y = BUCKET_INDEX(key_hash);
  Bucket<T> *target = bucket + y;
  //第63个桶的邻居是第0个桶`
  Bucket<T> *neighbor = bucket + ((y + 1) & bucketMask);
  //对target和neighbor都加锁
  target->get_lock();
  if (!neighbor->try_get_lock()) {
    target->release_lock();
    return -2;
  }

  //检验段，old_sa
  auto old_sa = *_dir;
  //得到MSB，MSB作为段索引
  auto x = (key_hash >> (8 * sizeof(key_hash) - old_sa->global_depth));
  //这一步主要是验证我们传入的想要插入的目录的段 和当前段是不是一样的
  //得到old_sa中的table的地址(好像比较56位即可)和this一不一致，question
  //question】
  if (reinterpret_cast<Table<T> *>(reinterpret_cast<uint64_t>(old_sa->_[x]) &
                                   tailMask) != this) {
    neighbor->release_lock();
    target->release_lock();
    return -2;
  }

  /*unique check, needs to check 2 hash table*/
  //唯一性检查
  auto ret =
      target->unique_check(meta_hash, key, neighbor, bucket + kNumBucket);
  if (ret == -1) {
    neighbor->release_lock();
    target->release_lock();
    return -3; /* duplicate insert*/
  }

  //如果b和b+1均满了的话 就进行displace操作
  if (((GET_COUNT(target->bitmap)) == kNumPairPerBucket) &&
      ((GET_COUNT(neighbor->bitmap)) == kNumPairPerBucket)) {
    Bucket<T> *next_neighbor = bucket + ((y + 2) & bucketMask);
    // Next displacement
    //获取接下来的锁，如果没获取的到的话，都释放掉
    if (!next_neighbor->try_get_lock()) {
      neighbor->release_lock();
      target->release_lock();
      return -2;
    }
    //将b+1的数据移动到b+2 同时插入
    auto ret =
        Next_displace(target, neighbor, next_neighbor, key, value, meta_hash);
    if (ret == 0) {
      return 0;
    }
    next_neighbor->release_lock();

    Bucket<T> *prev_neighbor;
    int prev_index;
    //第0个桶的前一个邻居是第63个桶
    if (y == 0) {
      prev_neighbor = bucket + kNumBucket - 1;
      prev_index = kNumBucket - 1;
    } else {
      prev_neighbor = bucket + y - 1;
      prev_index = y - 1;
    }
    if (!prev_neighbor->try_get_lock()) {
      target->release_lock();
      neighbor->release_lock();
      return -2;
    }

    ret = Prev_displace(target, prev_neighbor, neighbor, key, value, meta_hash);
    if (ret == 0) {
      return 0;
    }
    //如果insert都满了的话 那就插入到stash中
    Bucket<T> *stash = bucket + kNumBucket;
    //先的到stash桶锁
    if (!stash->try_get_lock()) {
      neighbor->release_lock();
      target->release_lock();
      prev_neighbor->release_lock();
      return -2;
    }
    //返回插入的结果，注意overflow的元数据作者没有持久化，好像
    ret = Stash_insert(target, neighbor, key, value, meta_hash, y & stashMask);

    //stash释放锁
    stash->release_lock();
    neighbor->release_lock();
    target->release_lock();
    prev_neighbor->release_lock();
    return ret;
  }

  /* the fp+bitmap are persisted after releasing the lock of one bucket but
   * still guarantee the correctness of avoidance of "use-before-flush" since
   * the search operation could only proceed only if both target bucket and
   * probe bucket are released
   */
  //load banlance策略
  if (GET_COUNT(target->bitmap) <= GET_COUNT(neighbor->bitmap)) {
    target->Insert(key, value, meta_hash, false);
    target->release_lock();
#ifdef PMEM
    Allocator::Persist(&target->bitmap, sizeof(target->bitmap));
#endif
    neighbor->release_lock();
  } else {
    neighbor->Insert(key, value, meta_hash, true);
    neighbor->release_lock();
#ifdef PMEM
    Allocator::Persist(&neighbor->bitmap, sizeof(neighbor->bitmap));
#endif
    target->release_lock();
  }
#ifdef COUNTING
  __sync_fetch_and_add(&number, 1);
#endif
  return 0;
}

template <class T>
void Table<T>::Insert4splitWithCheck(T key, Value_t value, size_t key_hash,
                                     uint8_t meta_hash) {
  auto y = BUCKET_INDEX(key_hash);
  Bucket<T> *target = bucket + y;
  Bucket<T> *neighbor = bucket + ((y + 1) & bucketMask);
  auto ret =
      target->unique_check(meta_hash, key, neighbor, bucket + kNumBucket);
  if (ret == -1) return;
  Bucket<T> *insert_target;
  bool probe = false;
  if (GET_COUNT(target->bitmap) <= GET_COUNT(neighbor->bitmap)) {
    insert_target = target;
  } else {
    insert_target = neighbor;
    probe = true;
  }

  /*some bucket may be overflowed?*/
  if (GET_COUNT(insert_target->bitmap) < kNumPairPerBucket) {
    insert_target->_[GET_COUNT(insert_target->bitmap)].key = key;
    insert_target->_[GET_COUNT(insert_target->bitmap)].value = value;
    insert_target->set_hash(GET_COUNT(insert_target->bitmap), meta_hash, probe);
#ifdef COUNTING
    ++number;
#endif
  } else {
    /*do the displacement or insertion in the stash*/
    Bucket<T> *next_neighbor = bucket + ((y + 2) & bucketMask);
    int displace_index;
    displace_index = neighbor->Find_org_displacement();
    if (((GET_COUNT(next_neighbor->bitmap)) != kNumPairPerBucket) &&
        (displace_index != -1)) {
      next_neighbor->Insert_with_noflush(
          neighbor->_[displace_index].key, neighbor->_[displace_index].value,
          neighbor->finger_array[displace_index], true);
      neighbor->unset_hash(displace_index);
      neighbor->Insert_displace_with_noflush(key, value, meta_hash,
                                             displace_index, true);
#ifdef COUNTING
      ++number;
#endif
      return;
    }
    Bucket<T> *prev_neighbor;
    int prev_index;
    if (y == 0) {
      prev_neighbor = bucket + kNumBucket - 1;
      prev_index = kNumBucket - 1;
    } else {
      prev_neighbor = bucket + y - 1;
      prev_index = y - 1;
    }

    displace_index = target->Find_probe_displacement();
    if (((GET_COUNT(prev_neighbor->bitmap)) != kNumPairPerBucket) &&
        (displace_index != -1)) {
      prev_neighbor->Insert_with_noflush(
          target->_[displace_index].key, target->_[displace_index].value,
          target->finger_array[displace_index], false);
      target->unset_hash(displace_index);
      target->Insert_displace_with_noflush(key, value, meta_hash,
                                           displace_index, false);
#ifdef COUNTING
      ++number;
#endif
      return;
    }

    Stash_insert(target, neighbor, key, value, meta_hash, y & stashMask);
  }
}

/*the insert needs to be perfectly balanced, not destory the power of balance*/
/*
 * split版本的插入，重新加入平衡策略，但是不涉及持久化操作
 * key:需要插入的key
 * value:需要插入的value
 * key_hash:哈希之后的key
 * meta_hash:指纹
 */
template <class T>
void Table<T>::Insert4split(T key, Value_t value, size_t key_hash,
                            uint8_t meta_hash) {
  //找到新段中，需要插入的数据的桶的位置
  auto y = BUCKET_INDEX(key_hash);
  //在新段中插入数据继续刚刚的load banlance策略
  Bucket<T> *target = bucket + y;
  Bucket<T> *neighbor = bucket + ((y + 1) & bucketMask);
  Bucket<T> *insert_target;
  bool probe = false;
  //看b和b+1这两个桶哪个数据少
  if (GET_COUNT(target->bitmap) <= GET_COUNT(neighbor->bitmap)) {
    insert_target = target;
  } else {
    insert_target = neighbor;
    probe = true;
  }

  /*some bucket may be overflowed?*/
  //如果没满的话，那么就插入。由于是个新段，Get_Count可以直接当做空闲位置的索引了，与Insert不同，这里不刷新数据
  if (GET_COUNT(insert_target->bitmap) < kNumPairPerBucket) {
    insert_target->_[GET_COUNT(insert_target->bitmap)].key = key;
    insert_target->_[GET_COUNT(insert_target->bitmap)].value = value;
    insert_target->set_hash(GET_COUNT(insert_target->bitmap), meta_hash, probe);
#ifdef COUNTING
    ++number;
#endif
  } else {
    //b和b+1都满了，执行displace操作，首先向后displace，看看能不能将b+1中的某个数据移动到b+2中
    /*do the displacement or insertion in the stash*/
    Bucket<T> *next_neighbor = bucket + ((y + 2) & bucketMask);
    int displace_index;
    displace_index = neighbor->Find_org_displacement();
    //b+2没满的话，往b+2中插入b+1的数据，需要修改对应的member
    if (((GET_COUNT(next_neighbor->bitmap)) != kNumPairPerBucket) &&
        (displace_index != -1)) {
      next_neighbor->Insert_with_noflush(
          neighbor->_[displace_index].key, neighbor->_[displace_index].value,
          neighbor->finger_array[displace_index], true);
      //b+1中删除这个数据，这里持久化删除了
      neighbor->unset_hash(displace_index);
      //往b+1中插入对应的数据，但是这里都不设计持久化操作
      neighbor->Insert_displace_with_noflush(key, value, meta_hash,
                                             displace_index, true);
#ifdef COUNTING
      ++number;
#endif
      return;
    }
    Bucket<T> *prev_neighbor;
    int prev_index;
    if (y == 0) {
      prev_neighbor = bucket + kNumBucket - 1;
      prev_index = kNumBucket - 1;
    } else {
      prev_neighbor = bucket + y - 1;
      prev_index = y - 1;
    }

    displace_index = target->Find_probe_displacement();
    if (((GET_COUNT(prev_neighbor->bitmap)) != kNumPairPerBucket) &&
        (displace_index != -1)) {
      prev_neighbor->Insert_with_noflush(
          target->_[displace_index].key, target->_[displace_index].value,
          target->finger_array[displace_index], false);
      //持久化删除的标记为
      target->unset_hash(displace_index);
      target->Insert_displace_with_noflush(key, value, meta_hash,
                                           displace_index, false);
#ifdef COUNTING
      ++number;
#endif
      return;
    }
    //这里持久化了overflowbucket的bitmap，y&stashMask这里唯一的好处是奇数桶stash从1->0，偶数桶从0->1
    Stash_insert(target, neighbor, key, value, meta_hash, y & stashMask);
  }
}



template <class T>
void Table<T>::Insert4merge(T key, Value_t value, size_t key_hash,
                            uint8_t meta_hash, bool unique_check_flag) {
  auto y = BUCKET_INDEX(key_hash);
  Bucket<T> *target = bucket + y;
  Bucket<T> *neighbor = bucket + ((y + 1) & bucketMask);

  if (unique_check_flag) {
    auto ret =
        target->unique_check(meta_hash, key, neighbor, bucket + kNumBucket);
    if (ret == -1) return;
  }

  Bucket<T> *insert_target;
  bool probe = false;
  if (GET_COUNT(target->bitmap) <= GET_COUNT(neighbor->bitmap)) {
    insert_target = target;
  } else {
    insert_target = neighbor;
    probe = true;
  }

  /*some bucket may be overflowed?*/
  if (GET_COUNT(insert_target->bitmap) < kNumPairPerBucket) {
    insert_target->Insert(key, value, meta_hash, probe);
#ifdef COUNTING
    ++number;
#endif
  } else {
    /*do the displacement or insertion in the stash*/
    Bucket<T> *next_neighbor = bucket + ((y + 2) & bucketMask);
    int displace_index;
    displace_index = neighbor->Find_org_displacement();
    if (((GET_COUNT(next_neighbor->bitmap)) != kNumPairPerBucket) &&
        (displace_index != -1)) {
      next_neighbor->Insert_with_noflush(
          neighbor->_[displace_index].key, neighbor->_[displace_index].value,
          neighbor->finger_array[displace_index], true);
      neighbor->unset_hash(displace_index);
      neighbor->Insert_displace_with_noflush(key, value, meta_hash,
                                             displace_index, true);
#ifdef COUNTING
      ++number;
#endif
      return;
    }
    Bucket<T> *prev_neighbor;
    int prev_index;
    if (y == 0) {
      prev_neighbor = bucket + kNumBucket - 1;
      prev_index = kNumBucket - 1;
    } else {
      prev_neighbor = bucket + y - 1;
      prev_index = y - 1;
    }

    displace_index = target->Find_probe_displacement();
    if (((GET_COUNT(prev_neighbor->bitmap)) != kNumPairPerBucket) &&
        (displace_index != -1)) {
      prev_neighbor->Insert_with_noflush(
          target->_[displace_index].key, target->_[displace_index].value,
          target->finger_array[displace_index], false);
      target->unset_hash(displace_index);
      target->Insert_displace_with_noflush(key, value, meta_hash,
                                           displace_index, false);
#ifdef COUNTING
      ++number;
#endif
      return;
    }

    Stash_insert(target, neighbor, key, value, meta_hash, y & stashMask);
  }
}

template <class T>
void Table<T>::HelpSplit(Table<T> *next_table) {
  size_t new_pattern = (pattern << 1) + 1;
  size_t old_pattern = pattern << 1;

  size_t key_hash;
  uint32_t invalid_array[kNumBucket + stashBucket];
  for (int i = 0; i < kNumBucket; ++i) {
    auto *curr_bucket = bucket + i;
    auto mask = GET_BITMAP(curr_bucket->bitmap);
    uint32_t invalid_mask = 0;
    for (int j = 0; j < kNumPairPerBucket; ++j) {
      if (CHECK_BIT(mask, j)) {
        if constexpr (std::is_pointer_v<T>) {
          auto curr_key = curr_bucket->_[j].key;
          key_hash = h(curr_key->key, curr_key->length);
        } else {
          key_hash = h(&(curr_bucket->_[j].key), sizeof(Key_t));
        }

        if ((key_hash >> (64 - local_depth - 1)) == new_pattern) {
          invalid_mask = invalid_mask | (1 << j);
          next_table->Insert4splitWithCheck(curr_bucket->_[j].key,
                                            curr_bucket->_[j].value, key_hash,
                                            curr_bucket->finger_array[j]);
#ifdef COUNTING
          number--;
#endif
        }
      }
    }
    invalid_array[i] = invalid_mask;
  }

  for (int i = 0; i < stashBucket; ++i) {
    auto *curr_bucket = bucket + kNumBucket + i;
    auto mask = GET_BITMAP(curr_bucket->bitmap);
    uint32_t invalid_mask = 0;
    for (int j = 0; j < kNumPairPerBucket; ++j) {
      if (CHECK_BIT(mask, j)) {
        if constexpr (std::is_pointer_v<T>) {
          auto curr_key = curr_bucket->_[j].key;
          key_hash = h(curr_key->key, curr_key->length);
        } else {
          key_hash = h(&(curr_bucket->_[j].key), sizeof(Key_t));
        }
        if ((key_hash >> (64 - local_depth - 1)) == new_pattern) {
          invalid_mask = invalid_mask | (1 << j);
          next_table->Insert4splitWithCheck(curr_bucket->_[j].key,
                                            curr_bucket->_[j].value, key_hash,
                                            curr_bucket->finger_array[j]);
          auto bucket_ix = BUCKET_INDEX(key_hash);
          auto org_bucket = bucket + bucket_ix;
          auto neighbor_bucket = bucket + ((bucket_ix + 1) & bucketMask);
          org_bucket->unset_indicator(curr_bucket->finger_array[j],
                                      neighbor_bucket, curr_bucket->_[j].key,
                                      i);
#ifdef COUNTING
          number--;
#endif
        }
      }
    }
    invalid_array[kNumBucket + i] = invalid_mask;
  }
  next_table->pattern = new_pattern;
  Allocator::Persist(&next_table->pattern, sizeof(next_table->pattern));
  pattern = old_pattern;
  Allocator::Persist(&pattern, sizeof(pattern));

#ifdef PMEM
  Allocator::Persist(next_table, sizeof(Table));
  size_t sumBucket = kNumBucket + stashBucket;
  for (int i = 0; i < sumBucket; ++i) {
    auto curr_bucket = bucket + i;
    curr_bucket->bitmap = curr_bucket->bitmap & (~(invalid_array[i] << 18)) &
                          (~(invalid_array[i] << 4));
    uint32_t count = __builtin_popcount(invalid_array[i]);
    curr_bucket->bitmap = curr_bucket->bitmap - count;
  }

  Allocator::Persist(this, sizeof(Table));
#endif
}

template <class T>
Table<T> *Table<T>::Split(size_t _key_hash) {
  //新段相对于旧段来说标号大1
  size_t new_pattern = (pattern << 1) + 1;
  size_t old_pattern = pattern << 1;

  //question：为什么不从0开始锁
  for (int i = 1; i < kNumBucket; ++i) {
    (bucket + i)->get_lock();
  }
  state = -2; /*means the start of the split process*/
  Allocator::Persist(&state, sizeof(state));
  //New 一个新段 一个New操作
  Table<T>::New(&next, local_depth + 1, next);
  //pmemobj_direct(next) 相当于得到next的地址，得到下一个指向下一个table的指针
  Table<T> *next_table = reinterpret_cast<Table<T> *>(pmemobj_direct(next));

  next_table->state = -2;
  Allocator::Persist(&next_table->state, sizeof(next_table->state));
  //现在再锁上第一个桶
  next_table->bucket
      ->get_lock(); /* get the first lock of the new bucket to avoid it
                 is operated(split or merge) by other threads*/
  size_t key_hash;
  
  //初始化数组32位，记录每个数组的元数据最重要的元数据即4B
  uint32_t invalid_array[kNumBucket + stashBucket];
  //开始修改标记位，正常桶
  for (int i = 0; i < kNumBucket; ++i) {
    auto *curr_bucket = bucket + i;
    //得到这个桶的bitmap
    auto mask = GET_BITMAP(curr_bucket->bitmap);
    //初始胡mask
    uint32_t invalid_mask = 0;
    //14个指纹，逐个比较，看看要迁移哪些数据
    for (int j = 0; j < kNumPairPerBucket; ++j) {
      //判断j处是否有数据
      if (CHECK_BIT(mask, j)) {
        //如果是指针的话
        if constexpr (std::is_pointer_v<T>) {
          //得到当前位置的key
          auto curr_key = curr_bucket->_[j].key;
          //对key进行rehash操作
          key_hash = h(curr_key->key, curr_key->length);
        } else {
          key_hash = h(&(curr_bucket->_[j].key), sizeof(Key_t));
        }
        //根据rehash之后的key的LSB确定在哪一个段，桶的位置没变，将需要迁移的条目找出来，迁移到新段
        if ((key_hash >> (64 - local_depth - 1)) == new_pattern) {
          //将标记位置为1
          invalid_mask = invalid_mask | (1 << j);
          //新段需要插入数据，旧段需要删除掉这个数据
          //旧段需要删除掉指纹，bitmap，counter--，membership--
          //新段插入即可
          next_table->Insert4split(
              curr_bucket->_[j].key, curr_bucket->_[j].value, key_hash,
              curr_bucket->finger_array[j]); /*this shceme may destory the
                                                balanced segment*/
                                             // curr_bucket->unset_hash(j);
#ifdef COUNTING
          number--;
#endif
        }
      }
    }
    //记录这个桶已经失效的数据(
    invalid_array[i] = invalid_mask;
  }

  //记下来迁移stash桶中的数据
  for (int i = 0; i < stashBucket; ++i) {
    //得到stash桶
    auto *curr_bucket = bucket + kNumBucket + i;
    //得到stash桶的bitmap
    auto mask = GET_BITMAP(curr_bucket->bitmap);
    //记录当前stash桶的一些数据
    uint32_t invalid_mask = 0;
    for (int j = 0; j < kNumPairPerBucket; ++j) {
      //判断j有没有位置
      if (CHECK_BIT(mask, j)) {
       //看存的是不是指针，然后看使用什么哈希函数
        if constexpr (std::is_pointer_v<T>) {
          auto curr_key = curr_bucket->_[j].key;
          key_hash = h(curr_key->key, curr_key->length);
        } else {
          key_hash = h(&(curr_bucket->_[j].key), sizeof(Key_t));
        }
        //查看LSB，看看是不是要转移
        if ((key_hash >> (64 - local_depth - 1)) == new_pattern) {
          invalid_mask = invalid_mask | (1 << j);
          next_table->Insert4split(
              curr_bucket->_[j].key, curr_bucket->_[j].value, key_hash,
              curr_bucket->finger_array[j]); /*this shceme may destory the
                                                balanced segment*/
          //查看这个key本应该属于哪一个桶
          auto bucket_ix = BUCKET_INDEX(key_hash);
          //得到这个桶
          auto org_bucket = bucket + bucket_ix;
          //得到neighbor
          auto neighbor_bucket = bucket + ((bucket_ix + 1) & bucketMask);
          //修改overflowmap都是没有持久化操作的
          org_bucket->unset_indicator(curr_bucket->finger_array[j],
                                      neighbor_bucket, curr_bucket->_[j].key,
                                      i);
#ifdef COUNTING
          number--;
#endif
        }
      }
    }
    invalid_array[kNumBucket + i] = invalid_mask;
  }
  //新分裂的段，持久化即可，首先持久化新分裂的本地深度 pattern
  next_table->pattern = new_pattern;
  Allocator::Persist(&next_table->pattern, sizeof(next_table->pattern));
  //再持久化旧段的pattern
  pattern = old_pattern;
  Allocator::Persist(&pattern, sizeof(pattern));

#ifdef PMEM
  //持久化整个新段
  Allocator::Persist(next_table, sizeof(Table));
  //然后删除旧段中分配到新段的数据，这里修改元数据即可，但是这里所有操作都不涉及fingerprint的修改
  size_t sumBucket = kNumBucket + stashBucket;
  for (int i = 0; i < sumBucket; ++i) {
    auto curr_bucket = bucket + i;
    //14bitmap + 14pointer bitmap + 4counter
    curr_bucket->bitmap = curr_bucket->bitmap & (~(invalid_array[i] << 18)) &
                          (~(invalid_array[i] << 4));
    uint32_t count = __builtin_popcount(invalid_array[i]);
    curr_bucket->bitmap = curr_bucket->bitmap - count;
  }
  //再持久化旧段
  Allocator::Persist(this, sizeof(Table));
#endif
  return next_table;
}

template <class T>
/*
 * 将b+1中的数据移动到b中
 */
void Table<T>::Merge(Table<T> *neighbor, bool unique_check_flag) {
  /*Restore the split/merge procedure*/
  //多了个唯一性检查而已。。
  if (unique_check_flag) {
    size_t key_hash;
    for (int i = 0; i < kNumBucket; ++i) {
      auto *curr_bucket = neighbor->bucket + i;
      auto mask = GET_BITMAP(curr_bucket->bitmap);
      for (int j = 0; j < kNumPairPerBucket; ++j) {
        if (CHECK_BIT(mask, j)) {
          if constexpr (std::is_pointer_v<T>) {
            auto curr_key = curr_bucket->_[j].key;
            key_hash = h(curr_key->key, curr_key->length);
          } else {
            key_hash = h(&(curr_bucket->_[j].key), sizeof(Key_t));
          }
          Insert4merge(curr_bucket->_[j].key, curr_bucket->_[j].value, key_hash,
                       curr_bucket->finger_array[j],
                       true); /*this shceme may destory
                           the balanced segment*/
        }
      }
    }

    for (int i = 0; i < stashBucket; ++i) {
      auto *curr_bucket = neighbor->bucket + kNumBucket + i;
      auto mask = GET_BITMAP(curr_bucket->bitmap);
      for (int j = 0; j < kNumPairPerBucket; ++j) {
        if (CHECK_BIT(mask, j)) {
          if constexpr (std::is_pointer_v<T>) {
            auto curr_key = curr_bucket->_[j].key;
            key_hash = h(curr_key->key, curr_key->length);
          } else {
            key_hash = h(&(curr_bucket->_[j].key), sizeof(Key_t));
          }
          Insert4merge(curr_bucket->_[j].key, curr_bucket->_[j].value, key_hash,
                       curr_bucket->finger_array[j]); /*this shceme may destory
                                                         the balanced segment*/
        }
      }
    }

  } else {
    size_t key_hash;
    for (int i = 0; i < kNumBucket; ++i) {
      auto *curr_bucket = neighbor->bucket + i;
      auto mask = GET_BITMAP(curr_bucket->bitmap);
      for (int j = 0; j < kNumPairPerBucket; ++j) {
        if (CHECK_BIT(mask, j)) {
          if constexpr (std::is_pointer_v<T>) {
            auto curr_key = curr_bucket->_[j].key;
            key_hash = h(curr_key->key, curr_key->length);
          } else {
            key_hash = h(&(curr_bucket->_[j].key), sizeof(Key_t));
          }

          Insert4merge(curr_bucket->_[j].key, curr_bucket->_[j].value, key_hash,
                       curr_bucket->finger_array[j]); /*this shceme may destory
                                                         the balanced segment*/
        }
      }
    }

    /*split the stash bucket, the stash must be full, right?*/
    for (int i = 0; i < stashBucket; ++i) {
      auto *curr_bucket = neighbor->bucket + kNumBucket + i;
      auto mask = GET_BITMAP(curr_bucket->bitmap);
      for (int j = 0; j < kNumPairPerBucket; ++j) {
        if (CHECK_BIT(mask, j)) {
          if constexpr (std::is_pointer_v<T>) {
            auto curr_key = curr_bucket->_[j].key;
            key_hash = h(curr_key->key, curr_key->length);
          } else {
            key_hash = h(&(curr_bucket->_[j].key), sizeof(Key_t));
          }
          Insert4merge(curr_bucket->_[j].key, curr_bucket->_[j].value, key_hash,
                       curr_bucket->finger_array[j]); /*this shceme may destory
                                                         the balanced segment*/
        }
      }
    }
  }
}

//哈希函数类
template <class T>
class Finger_EH : public Hash<T> {
 public:
  Finger_EH(void);
  Finger_EH(size_t, PMEMobjpool *_pool);
  ~Finger_EH(void);
  //插入函数，插入key和value
  inline int Insert(T key, Value_t value);
  //这个insert不太懂，对于epoch还需要进行理解
  int Insert(T key, Value_t value, bool);
  //删除函数
  inline bool Delete(T);
  //和epoch又有关系了
  bool Delete(T, bool);
  //查找函数
  inline Value_t Get(T);
  Value_t Get(T key, bool is_in_epoch);
  
  void TryMerge(size_t key_hash);
  void Directory_Doubling(int x, Table<T> *new_b, Table<T> *old_b);
  void Directory_Merge_Update(Directory<T> *_sa, uint64_t key_hash,
                              Table<T> *left_seg);
  void Directory_Update(Directory<T> *_sa, int x, Table<T> *new_b,
                        Table<T> *old_b);
  void Halve_Directory();
  int FindAnyway(T key);
  //正常关机
  void ShutDown() {
    clean = true;
    Allocator::Persist(&clean, sizeof(clean));
  }
  void getNumber() {
    std::cout << "The size of the bucket is " << sizeof(struct Bucket<T>) << std::endl;
    size_t _count = 0;
    size_t seg_count = 0;
    Directory<T> *seg = dir;
    Table<T> **dir_entry = seg->_;
    Table<T> *ss;
    auto global_depth = seg->global_depth;
    size_t depth_diff;
    int capacity = pow(2, global_depth);
    for (int i = 0; i < capacity;) {
      ss = reinterpret_cast<Table<T> *>(
          reinterpret_cast<uint64_t>(dir_entry[i]) & tailMask);
      depth_diff = global_depth - ss->local_depth;
      _count += ss->number;
      seg_count++;
      i += pow(2, depth_diff);
    }

    ss = reinterpret_cast<Table<T> *>(reinterpret_cast<uint64_t>(dir_entry[0]) &
                                      tailMask);
    uint64_t verify_seg_count = 1;
    while (!OID_IS_NULL(ss->next)) {
      verify_seg_count++;
      ss = reinterpret_cast<Table<T> *>(pmemobj_direct(ss->next));
    }
    std::cout << "seg_count = " << seg_count << std::endl;
    std::cout << "verify_seg_count = " << verify_seg_count << std::endl;
#ifdef COUNTING
    std::cout << "#items = " << _count << std::endl;
    std::cout << "load_factor = " <<
           (double)_count / (seg_count * kNumPairPerBucket * (kNumBucket + 2)) << std::endl;
    std::cout << "Raw_Space: ",
           (double)(_count * 16) / (seg_count * sizeof(Table<T>)) << std::endl;
#endif
  }

  void recoverTable(Table<T> **target_table, size_t, size_t, Directory<T> *);
  //用于目录的修理，clean=true表示正常关机
  void Recovery();

  inline int Test_Directory_Lock_Set(void) {
    uint32_t v = __atomic_load_n(&lock, __ATOMIC_ACQUIRE);
    return v & lockSet;
  }

  //这里并不加锁，只是增加一次版本号，lock的第一位依然为0
  inline bool try_get_directory_read_lock(){
    uint32_t v = __atomic_load_n(&lock, __ATOMIC_ACQUIRE);
    uint32_t old_value = v & lockMask;
    auto new_value = ((v & lockMask) + 1) & lockMask;
    return CAS(&lock, &old_value, new_value);
  }

  //版本号++
  inline void release_directory_read_lock(){
    SUB(&lock, 1);
  }

  void Lock_Directory(){
    //加锁
    uint32_t v = __atomic_load_n(&lock, __ATOMIC_ACQUIRE);
    //后31位当做版本号
    uint32_t old_value = v & lockMask;
    //第一位当作锁
    uint32_t new_value = old_value | lockSet;
    //CAS原语 直到成功
    while (!CAS(&lock, &old_value, new_value)) {
      old_value = old_value & lockMask;
      new_value = old_value | lockSet;
    }

    //wait until the readers all exit the critical section
    v = __atomic_load_n(&lock, __ATOMIC_ACQUIRE);
    //当v没有被置为0的话 一直读取v
    while(v & lockMask){
      v = __atomic_load_n(&lock, __ATOMIC_ACQUIRE);
    }
  }

  // just set the lock as 0
  void Unlock_Directory(){
    __atomic_store_n(&lock, 0, __ATOMIC_RELEASE);    
  }

  Directory<T> *dir;
  uint32_t lock; // the MSB is the lock bit; remaining bits are used as the counter
  uint64_t
      crash_version; /*when the crash version equals to 0Xff => set the crash
                        version as 0, set the version of all entries as 1*/
  bool clean;
  PMEMobjpool *pool_addr;
  /* directory allocation will write to here first,
   * in oder to perform safe directory allocation
   * */
  PMEMoid back_dir;//之前的目录

};

template <class T>
Finger_EH<T>::Finger_EH(size_t initCap, PMEMobjpool *_pool) {
  pool_addr = _pool;
  Directory<T>::New(&back_dir, initCap, 0);
  dir = reinterpret_cast<Directory<T> *>(pmemobj_direct(back_dir));
  back_dir = OID_NULL;
  lock = 0;
  crash_version = 0;
  clean = false;
  PMEMoid ptr;

  /*FIXME: make the process of initialization crash consistent*/
  Table<T>::New(&ptr, dir->global_depth, OID_NULL);
  dir->_[initCap - 1] = (Table<T> *)pmemobj_direct(ptr);
  dir->_[initCap - 1]->pattern = initCap - 1;
  dir->_[initCap - 1]->state = 0;
  /* Initilize the Directory*/
  for (int i = initCap - 2; i >= 0; --i) {
    Table<T>::New(&ptr, dir->global_depth, ptr);
    dir->_[i] = (Table<T> *)pmemobj_direct(ptr);
    dir->_[i]->pattern = i;
    dir->_[i]->state = 0;
  }
  dir->depth_count = initCap;
}

template <class T>
Finger_EH<T>::Finger_EH() {
  std::cout << "Reinitialize up" << std::endl;
}

template <class T>
Finger_EH<T>::~Finger_EH(void) {
  // TO-DO
}

template <class T>
void Finger_EH<T>::Halve_Directory() {
  std::cout << "Begin::Directory_Halving towards " <<  dir->global_depth << std::endl;
  auto d = dir->_;

  Directory<T> *new_dir;
#ifdef PMEM
  Directory<T>::New(&back_dir, pow(2, dir->global_depth - 1), dir->version + 1);
  new_dir = reinterpret_cast<Directory<T> *>(pmemobj_direct(back_dir));
#else
  Directory<T>::New(&new_dir, pow(2, dir->global_depth - 1), dir->version + 1);
#endif

  auto _dir = new_dir->_;
  new_dir->depth_count = 0;
  auto capacity = pow(2, new_dir->global_depth);
  bool skip = false;
  for (int i = 0; i < capacity; ++i) {
    _dir[i] = d[2 * i];
    assert(d[2 * i] == d[2 * i + 1]);
    if (!skip) {
      if ((_dir[i]->local_depth == (dir->global_depth - 1)) &&
          (_dir[i]->state != -2)) {
        if (_dir[i]->state != -1) {
          new_dir->depth_count += 1;
        } else {
          skip = true;
        }
      }
    } else {
      skip = false;
    }
  }

#ifdef PMEM
  Allocator::Persist(new_dir,
                     sizeof(Directory<T>) + sizeof(uint64_t) * capacity);
  auto reserve_item = Allocator::ReserveItem();
  TX_BEGIN(pool_addr) {
    pmemobj_tx_add_range_direct(reserve_item, sizeof(*reserve_item));
    pmemobj_tx_add_range_direct(&dir, sizeof(dir));
    pmemobj_tx_add_range_direct(&back_dir, sizeof(back_dir));
    Allocator::Free(reserve_item, dir);
    dir = new_dir;
    back_dir = OID_NULL;
  }
  TX_ONABORT {
    std::cout << "TXN fails during halvling directory" << std::endl;
  }
  TX_END
#else
  dir = new_dir;
#endif
  std::cout << "End::Directory_Halving towards " << dir->global_depth << std::endl;
}

//目录扩展
template <class T>
void Finger_EH<T>::Directory_Doubling(int x, Table<T> *new_b, Table<T> *old_b) {
  Table<T> **d = dir->_;
  auto global_depth = dir->global_depth;
  std::cout << "Directory_Doubling towards " << global_depth + 1 << std::endl;

  auto capacity = pow(2, global_depth);
  //目录版本++，容量变大一倍
  Directory<T>::New(&back_dir, 2 * capacity, dir->version + 1);
  //新的
  Directory<T> *new_sa =
      reinterpret_cast<Directory<T> *>(pmemobj_direct(back_dir));
  //变大一倍之后的目录
  auto dd = new_sa->_;
  //复制之前的段
  for (unsigned i = 0; i < capacity; ++i) {
    dd[2 * i] = d[i];
    dd[2 * i + 1] = d[i];
  }
  //扩展后的目录指向新的，并且将crash_version修改为对的
  dd[2 * x + 1] = reinterpret_cast<Table<T> *>(
      reinterpret_cast<uint64_t>(new_b) | crash_version);
  //question depth_count的作用是啥
  new_sa->depth_count = 2;

#ifdef PMEM
  //持久化新段，New只是在DRAM中将内存分配了出来，Persist将内存保存在内存中
  Allocator::Persist(new_sa,
                     sizeof(Directory<T>) + sizeof(uint64_t) * 2 * capacity);
  auto reserve_item = Allocator::ReserveItem();
  ++merge_time;
  auto old_dir = dir;
  //将旧段的目录释放掉，同时dir交换掉
  TX_BEGIN(pool_addr) {
    pmemobj_tx_add_range_direct(reserve_item, sizeof(*reserve_item));
    pmemobj_tx_add_range_direct(&dir, sizeof(dir));
    pmemobj_tx_add_range_direct(&back_dir, sizeof(back_dir));
    pmemobj_tx_add_range_direct(&old_b->local_depth,
                                sizeof(old_b->local_depth));
    //旧段的本地深度++
    old_b->local_depth += 1;
    Allocator::Free(reserve_item, dir);
    /*Swap the memory addr between new directory and old directory*/
    //现在dir等于新分配出来的段，
    dir = new_sa;
    back_dir = OID_NULL;
  }
  TX_ONABORT {
    std::cout << "TXN fails during doubling directory" << std::endl;
  }
  TX_END

#else
  dir = new_sa;
#endif
}

template <class T>
void Finger_EH<T>::Directory_Update(Directory<T> *_sa, int x, Table<T> *new_b,
                                    Table<T> *old_b) {
  //_sa->_ table*[0]包含0个指针的table*
  Table<T> **dir_entry = _sa->_;
  auto global_depth = _sa->global_depth;
  unsigned depth_diff = global_depth - new_b->local_depth;
  //如果新段local_depth和目录的全局深度一样的话，只需要将x或者x+1中的某一个指向新产生的桶即可
  if (depth_diff == 0) {
    //这里打个很简单的比方，目录为000 001 现在要进行分割了，现在二者指向同一个段。
    //当000进行分割的话 那么dir_entry[001]指向新的段即可即dir_entry[001]=new table;
    //当001分割的时候，dir_entry[000]指向旧段不变，只需要让dir_entry[001]指向new table
    if (x % 2 == 0) {
      TX_BEGIN(pool_addr) {
        //分配新的内存
        pmemobj_tx_add_range_direct(&dir_entry[x + 1], sizeof(Table<T> *));
        pmemobj_tx_add_range_direct(&old_b->local_depth,
                                    sizeof(old_b->local_depth));
        //这个目录对应的table就等于new table
        dir_entry[x + 1] = reinterpret_cast<Table<T> *>(
            reinterpret_cast<uint64_t>(new_b) | crash_version);
        //旧段的本地深度++
        old_b->local_depth += 1;
      }
      TX_ONABORT { std::cout << "Error for update txn" << std::endl; }
      TX_END
    } else {
      TX_BEGIN(pool_addr) {
        pmemobj_tx_add_range_direct(&dir_entry[x], sizeof(Table<T> *));
        pmemobj_tx_add_range_direct(&old_b->local_depth,
                                    sizeof(old_b->local_depth));
        dir_entry[x] = reinterpret_cast<Table<T> *>(
            reinterpret_cast<uint64_t>(new_b) | crash_version);
        //旧段本地深度++
        old_b->local_depth += 1;
      }
      TX_ONABORT { std::cout << "Error for update txn" << std::endl; }
      TX_END
    }
#ifdef COUNTING
    __sync_fetch_and_add(&_sa->depth_count, 2);
#endif
    //当全局深度-本地深度不等于0的时候 这时候需要进行的是 前一半的目录指向旧桶，后一半的目录指向新桶
  } else {
    //chunk_size指的是开始有多少的目录对应同一个目录
    int chunk_size = pow(2, global_depth - (new_b->local_depth - 1));
    //找到最左边的端点
    x = x - (x % chunk_size);
    
    int base = chunk_size / 2;
    //这里不是分配内存的操作，而是将x+base这一块的内存先拷贝出来，因为要修改指针，这一部分的指针指向新段
    TX_BEGIN(pool_addr) {
      pmemobj_tx_add_range_direct(&dir_entry[x + base],
                                  sizeof(Table<T> *) * base);
      pmemobj_tx_add_range_direct(&old_b->local_depth,
                                  sizeof(old_b->local_depth));
      //从右到左更新
      for (int i = base - 1; i >= 0; --i) {
        //指针的前1B作为崩溃的版本号
        dir_entry[x + base + i] = reinterpret_cast<Table<T> *>(
            reinterpret_cast<uint64_t>(new_b) | crash_version);
      }
      old_b->local_depth += 1;
    }
    TX_ONABORT { std::cout << "Error for update txn" << std::endl; }
    TX_END
  }
  // printf("Done!directory update for %d\n", x);
}

template <class T>
void Finger_EH<T>::Directory_Merge_Update(Directory<T> *_sa, uint64_t key_hash,
                                          Table<T> *left_seg) {
  Table<T> **dir_entry = _sa->_;
  auto global_depth = _sa->global_depth;
  auto x = (key_hash >> (8 * sizeof(key_hash) - global_depth));
  uint64_t chunk_size = pow(2, global_depth - (left_seg->local_depth));
  auto left = x - (x % chunk_size);
  auto right = left + chunk_size / 2;

  for (int i = right; i < right + chunk_size / 2; ++i) {
    dir_entry[i] = left_seg;
    Allocator::Persist(&dir_entry[i], sizeof(uint64_t));
  }

  if ((left_seg->local_depth + 1) == global_depth) {
    SUB(&_sa->depth_count, 2);
  }
}

template <class T>
void Finger_EH<T>::recoverTable(Table<T> **target_table, size_t key_hash,
                                size_t x, Directory<T> *old_sa) {
  /*Set the lockBit to ahieve the mutal exclusion of the recover process*/
  //得到段落的首地址
  auto dir_entry = old_sa->_;
  //snapshot 得到Table*的地址
  uint64_t snapshot = (uint64_t)*target_table;
  //得到指定的table 为什么不直接传进来
  Table<T> *target = (Table<T> *)(snapshot & tailMask);
  //互斥锁
  //If the mutex is already locked, pthread_mutex_trylock() will not block waiting for the mutex,
  // but will return an error. If this is the first use of the mutex since the opening of the pool pop,
  // the mutex is automatically reinitialized and then locked.
  if (pmemobj_mutex_trylock(pool_addr, &target->lock_bit) != 0) {
    return;
  }
  //恢复段中bucket的stash元数据，并且删除重复的数据
  target->recoverMetadata();
  //继续桶之前的操作
  if (target->state != 0) {
    //得到段的pattern，相当于标号
    target->pattern = key_hash >> (8 * sizeof(key_hash) - target->local_depth);
    //根据本地深度计算出段的标号，并且持久化
    Allocator::Persist(&target->pattern, sizeof(target->pattern));
    //得到这个段的下一个段
    Table<T> *next_table = (Table<T> *)pmemobj_direct(target->next);
    //如果目标端当前的状态为分裂状态
    if (target->state == -2) {
      //下一个段为新段状态
      if (next_table->state == -3) {
        /*Help finish the split operation*/
        //删除新段的重复数据
        next_table->recoverMetadata();
        //新增唯一性检测
        target->HelpSplit(next_table);
        //锁住整个目录
        Lock_Directory();
        //得到段的位置MSB
        auto x = (key_hash >> (8 * sizeof(key_hash) - dir->global_depth));
        //如果target的本地深度小于全局深度的话 那么只需要段分割即可
        if (target->local_depth < dir->global_depth) {
          Directory_Update(dir, x, next_table, target);
        } else {
          //如果target的本地深度大于全局深度的话，那么就需要目录分割
          Directory_Doubling(x, next_table, target);
        }
        //解锁 版本号
        Unlock_Directory();
        /*release the lock for the target bucket and the new bucket*/
        //分割完成你也是正常段了
        next_table->state = 0;
        //将状态持久化
        Allocator::Persist(&next_table->state, sizeof(int));
      }
    } else if (target->state == -1) {
      //比较目录标号，如果正在合并的话，继续合并操作
      if (next_table->pattern == ((target->pattern << 1) + 1)) {
        target->Merge(next_table, true);
        Allocator::Persist(target, sizeof(Table<T>));
        target->next = next_table->next;
        Allocator::Free(next_table);
      }
    }
    //恢复完成，也是一个正常段了
    target->state = 0;
    //持久化state
    Allocator::Persist(&target->state, sizeof(int));
  }
  //等于0的话，将标志clear
  /*Compute for all entries and clear the dirty bit*/
  int chunk_size = pow(2, old_sa->global_depth - target->local_depth);
  x = x - (x % chunk_size);
  //让它的carsh_version等于崩溃之前的version
  for (int i = x; i < (x + chunk_size); ++i) {
    dir_entry[i] = reinterpret_cast<Table<T> *>(
        (reinterpret_cast<uint64_t>(dir_entry[i]) & tailMask) | crash_version);
  }
  //question:前面的这一层for循环应该包括他了吧，为啥它这里还写一遍
  *target_table = reinterpret_cast<Table<T> *>(
      reinterpret_cast<uint64_t>(target) | crash_version);
}

template <class T>
void Finger_EH<T>::Recovery() {
  /*scan the directory, set the clear bit, and also set the dirty bit in the
   * segment to indicate that this segment is clean*/
  if (clean) {
    clean = false;
    return;
  }
  Allocator::EpochRecovery();
  lock = 0;
  /*first check the back_dir log*/
  if (!OID_IS_NULL(back_dir)) {
    pmemobj_free(&back_dir);
  }

  auto dir_entry = dir->_;
  int length = pow(2, dir->global_depth);
  //crash_version 0xFF
  crash_version = ((crash_version >> 56) + 1) << 56;
  if (crash_version == 0) {
    uint64_t set_one = 1UL << 56;
    for (int i = 0; i < length; ++i) {
      uint64_t snapshot = (uint64_t)dir_entry[i];
      //将目录的crashversion置为1
      dir_entry[i] =
          reinterpret_cast<Table<T> *>((snapshot & tailMask) | set_one);
    }
    Allocator::Persist(dir_entry, sizeof(uint64_t) * length);
  }
}

template <class T>
int Finger_EH<T>::Insert(T key, Value_t value, bool is_in_epoch) {
  if (!is_in_epoch) {
    auto epoch_guard = Allocator::AquireEpochGuard();
    return Insert(key, value);
  }
  return Insert(key, value);
}

template <class T>
int Finger_EH<T>::Insert(T key, Value_t value) {
  uint64_t key_hash;
  if constexpr (std::is_pointer_v<T>) {
    key_hash = h(key->key, key->length);
  } else {
    key_hash = h(&key, sizeof(key));
  }
  //指纹
  auto meta_hash = ((uint8_t)(key_hash & kMask));  // the last 8 bits
RETRY:
  //old_sa
  auto old_sa = dir;
  //MSB查看段在哪个位置
  auto x = (key_hash >> (8 * sizeof(key_hash) - old_sa->global_depth));
  //dir_entry Table*[0],是一个指针数组
  auto dir_entry = old_sa->_;
  //dir_entry[x] 是一个Table* 准确来说是个指针,是一个地址
  //target就是dir_entry[x]
  Table<T> *target = reinterpret_cast<Table<T> *>(
      reinterpret_cast<uint64_t>(dir_entry[x]) & tailMask);

  //前1B用作仿作crash_version
  if ((reinterpret_cast<uint64_t>(dir_entry[x]) & headerMask) !=
      crash_version) {
    //恢复段
    /*
     * dir_entry[x] 需要修复的段
     * key_hash 哈希之后的值
     * x:段标号
     * old_sa 所在目录
     */
    //&dir_entry[x] 代表Table** 第一个*解引用之后是一个指向Table的地址
    //这里传入的MSB是根据目录的全局深度计算而来
    //第二个*解引用就是地址了
    recoverTable(&dir_entry[x], key_hash, x, old_sa);
    goto RETRY;
  }
  //段级别的插入，插入数据
  auto ret = target->Insert(key, value, key_hash, meta_hash, &dir);
  //
  if(ret == -3){ /*duplicate insertinsert, insertion failure*/
    return -1;
  }
  //ret==-1 需要段分割或者目录分割了，桶都满了
  if (ret == -1) {
    if (!target->bucket->try_get_lock()) {
      goto RETRY;
    }

    /*verify procedure*/
    auto old_sa = dir;
    auto x = (key_hash >> (8 * sizeof(key_hash) - old_sa->global_depth));
    //访问table的时候需要进行验证
    if (reinterpret_cast<Table<T> *>(reinterpret_cast<uint64_t>(old_sa->_[x]) &
                                     tailMask) != target) /* verify process*/
    {
      //验证失败的话，释放锁继续retry
      target->bucket->release_lock();
      goto RETRY;
    }
    //满了 对段进行分裂
    auto new_b =
        target->Split(key_hash); /* also needs the verify..., and we use try
                                    lock for this rather than the spin lock*/
    /* update directory*/
    /*更新目录 插入新的数据*/
  REINSERT:
    old_sa = dir;
    dir_entry = old_sa->_;
    x = (key_hash >> (8 * sizeof(key_hash) - old_sa->global_depth));
    //要插入的段的本地深度小于全局深度 只需要进行段分割即可
    if (target->local_depth < old_sa->global_depth) {
      //查看是否得到了锁
      if(!try_get_directory_read_lock()){
        goto REINSERT;
      }
      //查看
      if (old_sa->version != dir->version) {
        // The directory has changed, thus need retry this update
        release_directory_read_lock();
        goto REINSERT;
      }

      Directory_Update(old_sa, x, new_b, target);
      release_directory_read_lock();
    } else {
      Lock_Directory();
      if (old_sa->version != dir->version) {
        Unlock_Directory();
        goto REINSERT;
      }
      Directory_Doubling(x, new_b, target);
      Unlock_Directory();
    }

    /*release the lock for the target bucket and the new bucket*/
    new_b->state = 0;
    Allocator::Persist(&new_b->state, sizeof(int));
    target->state = 0;
    Allocator::Persist(&target->state, sizeof(int));

    Bucket<T> *curr_bucket;
    for (int i = 0; i < kNumBucket; ++i) {
      curr_bucket = target->bucket + i;
      curr_bucket->release_lock();
    }
    curr_bucket = new_b->bucket;
    curr_bucket->release_lock();
    goto RETRY;
  } else if (ret == -2) {
    goto RETRY;
  }
  //作者通过将第一个桶锁住 表示这个段正在分割（为什么不根据state呢）
  return 0;
}

template <class T>
Value_t Finger_EH<T>::Get(T key, bool is_in_epoch) {
  if (!is_in_epoch) {
    auto epoch_guard = Allocator::AquireEpochGuard();
    return Get(key);
  }
  uint64_t key_hash;
  if constexpr (std::is_pointer_v<T>) {
    key_hash = h(key->key, key->length);
  } else {
    key_hash = h(&key, sizeof(key));
  }
  auto meta_hash = ((uint8_t)(key_hash & kMask));  // the last 8 bits
RETRY:
  auto old_sa = dir;
  auto x = (key_hash >> (8 * sizeof(key_hash) - old_sa->global_depth));
  auto y = BUCKET_INDEX(key_hash);
  auto dir_entry = old_sa->_;
  auto old_entry = dir_entry[x];
  Table<T> *target = reinterpret_cast<Table<T> *>(
      reinterpret_cast<uint64_t>(old_entry) & tailMask);

  if ((reinterpret_cast<uint64_t>(old_entry) & headerMask) != crash_version) {
    recoverTable(&dir_entry[x], key_hash, x, old_sa);
    goto RETRY;
  }

  Bucket<T> *target_bucket = target->bucket + y;
  Bucket<T> *neighbor_bucket = target->bucket + ((y + 1) & bucketMask);

  uint32_t old_version =
      __atomic_load_n(&target_bucket->version_lock, __ATOMIC_ACQUIRE);
  uint32_t old_neighbor_version =
      __atomic_load_n(&neighbor_bucket->version_lock, __ATOMIC_ACQUIRE);

  if ((old_version & lockSet) || (old_neighbor_version & lockSet)) {
    goto RETRY;
  }

  /*verification procedure*/
  old_sa = dir;
  x = (key_hash >> (8 * sizeof(key_hash) - old_sa->global_depth));
  if (old_sa->_[x] != old_entry) {
    goto RETRY;
  }

  auto ret = target_bucket->check_and_get(meta_hash, key, false);
  if (target_bucket->test_lock_version_change(old_version)) {
    goto RETRY;
  }
  if (ret != NONE) {
    return ret;
  }

  /*no need for verification procedure, we use the version number of
   * target_bucket to test whether the bucket has ben spliteted*/
  ret = neighbor_bucket->check_and_get(meta_hash, key, true);
  if (neighbor_bucket->test_lock_version_change(old_neighbor_version)) {
    goto RETRY;
  }
  if (ret != NONE) {
    return ret;
  }

  if (target_bucket->test_stash_check()) {
    auto test_stash = false;
    if (target_bucket->test_overflow()) {
      /*this only occur when the bucket has more key-values than 10 that are
       * overfloed int he shared bucket area, therefore it needs to search in
       * the extra bucket*/
      test_stash = true;
    } else {
      /*search in the original bucket*/
      int mask = target_bucket->overflowBitmap & overflowBitmapMask;
      if (mask != 0) {
        for (int i = 0; i < 4; ++i) {
          if (CHECK_BIT(mask, i) &&
              (target_bucket->finger_array[14 + i] == meta_hash) &&
              (((1 << i) & target_bucket->overflowMember) == 0)) {
            Bucket<T> *stash =
                target->bucket + kNumBucket +
                ((target_bucket->overflowIndex >> (i * 2)) & stashMask);
            auto ret = stash->check_and_get(meta_hash, key, false);
            if (ret != NONE) {
              if (target_bucket->test_lock_version_change(old_version)) {
                goto RETRY;
              }
              return ret;
            }
          }
        }
      }

      mask = neighbor_bucket->overflowBitmap & overflowBitmapMask;
      if (mask != 0) {
        for (int i = 0; i < 4; ++i) {
          if (CHECK_BIT(mask, i) &&
              (neighbor_bucket->finger_array[14 + i] == meta_hash) &&
              (((1 << i) & neighbor_bucket->overflowMember) != 0)) {
            Bucket<T> *stash =
                target->bucket + kNumBucket +
                ((neighbor_bucket->overflowIndex >> (i * 2)) & stashMask);
            auto ret = stash->check_and_get(meta_hash, key, false);
            if (ret != NONE) {
              if (target_bucket->test_lock_version_change(old_version)) {
                goto RETRY;
              }
              return ret;
            }
          }
        }
      }
      goto FINAL;
    }
  TEST_STASH:
    if (test_stash == true) {
      for (int i = 0; i < stashBucket; ++i) {
        Bucket<T> *stash =
            target->bucket + kNumBucket + ((i + (y & stashMask)) & stashMask);
        auto ret = stash->check_and_get(meta_hash, key, false);
        if (ret != NONE) {
          if (target_bucket->test_lock_version_change(old_version)) {
            goto RETRY;
          }
          return ret;
        }
      }
    }
  }
FINAL:
  return NONE;
}

template <class T>
Value_t Finger_EH<T>::Get(T key) {
  uint64_t key_hash;
  if constexpr (std::is_pointer_v<T>) {
    key_hash = h(key->key, key->length);
  } else {
    key_hash = h(&key, sizeof(key));
  }
  auto meta_hash = ((uint8_t)(key_hash & kMask));  // the last 8 bits
RETRY:
  auto old_sa = dir;
  auto x = (key_hash >> (8 * sizeof(key_hash) - old_sa->global_depth));
  auto y = BUCKET_INDEX(key_hash);
  auto dir_entry = old_sa->_;
  auto old_entry = dir_entry[x];
  Table<T> *target = reinterpret_cast<Table<T> *>(
      reinterpret_cast<uint64_t>(old_entry) & tailMask);

  if ((reinterpret_cast<uint64_t>(old_entry) & headerMask) != crash_version) {
    recoverTable(&dir_entry[x], key_hash, x, old_sa);
    goto RETRY;
  }

  Bucket<T> *target_bucket = target->bucket + y;
  Bucket<T> *neighbor_bucket = target->bucket + ((y + 1) & bucketMask);

  //被锁住的话retry
  uint32_t old_version =
      __atomic_load_n(&target_bucket->version_lock, __ATOMIC_ACQUIRE);
  uint32_t old_neighbor_version =
      __atomic_load_n(&neighbor_bucket->version_lock, __ATOMIC_ACQUIRE);

  if ((old_version & lockSet) || (old_neighbor_version & lockSet)) {
    goto RETRY;
  }

  /*verification procedure*/
  old_sa = dir;
  x = (key_hash >> (8 * sizeof(key_hash) - old_sa->global_depth));
  if (old_sa->_[x] != old_entry) {
    goto RETRY;
  }

  auto ret = target_bucket->check_and_get(meta_hash, key, false);
  //查看读出来的时候 版本号变了没
  if (target_bucket->test_lock_version_change(old_version)) {
    goto RETRY;
  }
  if (ret != NONE) {
    return ret;
  }
  //target没找到的话，在neighbor中找
  ret = neighbor_bucket->check_and_get(meta_hash, key, true);
  if (neighbor_bucket->test_lock_version_change(old_neighbor_version)) {
    goto RETRY;
  }
  if (ret != NONE) {
    return ret;
  }
  //开始在stash中找，类似于delete。
  if (target_bucket->test_stash_check()) {
    auto test_stash = false;
    if (target_bucket->test_overflow()) {
      /*this only occur when the bucket has more key-values than 10 that are
       * overfloed int he shared bucket area, therefore it needs to search in
       * the extra bucket*/
      test_stash = true;
    } else {
      /*search in the original bucket*/
      int mask = target_bucket->overflowBitmap & overflowBitmapMask;
      if (mask != 0) {
        for (int i = 0; i < 4; ++i) {
          if (CHECK_BIT(mask, i) &&
              (target_bucket->finger_array[14 + i] == meta_hash) &&
              (((1 << i) & target_bucket->overflowMember) == 0)) {
            Bucket<T> *stash =
                target->bucket + kNumBucket +
                ((target_bucket->overflowIndex >> (i * 2)) & stashMask);
            auto ret = stash->check_and_get(meta_hash, key, false);
            if (ret != NONE) {
              if (target_bucket->test_lock_version_change(old_version)) {
                goto RETRY;
              }
              return ret;
            }
          }
        }
      }

      mask = neighbor_bucket->overflowBitmap & overflowBitmapMask;
      if (mask != 0) {
        for (int i = 0; i < 4; ++i) {
          if (CHECK_BIT(mask, i) &&
              (neighbor_bucket->finger_array[14 + i] == meta_hash) &&
              (((1 << i) & neighbor_bucket->overflowMember) != 0)) {
            Bucket<T> *stash =
                target->bucket + kNumBucket +
                ((neighbor_bucket->overflowIndex >> (i * 2)) & stashMask);
            auto ret = stash->check_and_get(meta_hash, key, false);
            if (ret != NONE) {
              if (target_bucket->test_lock_version_change(old_version)) {
                goto RETRY;
              }
              return ret;
            }
          }
        }
      }
      goto FINAL;
    }
  TEST_STASH:
    if (test_stash == true) {
      for (int i = 0; i < stashBucket; ++i) {
        Bucket<T> *stash =
            target->bucket + kNumBucket + ((i + (y & stashMask)) & stashMask);
        auto ret = stash->check_and_get(meta_hash, key, false);
        if (ret != NONE) {
          if (target_bucket->test_lock_version_change(old_version)) {
            goto RETRY;
          }
          return ret;
        }
      }
    }
  }
FINAL:
  return NONE;
}

template <class T>
void Finger_EH<T>::TryMerge(size_t key_hash) {
  /*Compute the left segment and right segment*/
  do {
    auto old_dir = dir;
    auto x = (key_hash >> (8 * sizeof(key_hash) - old_dir->global_depth));
    //这个哈希所在的段
    auto target = old_dir->_[x];
    //chunk_size指的是合并所涉及的段的数目
    int chunk_size = pow(2, old_dir->global_depth - (target->local_depth - 1));
    assert(chunk_size >= 2);
    //把右边的段给左边
    int left = x - (x % chunk_size);
    int right = left + chunk_size / 2;
    auto left_seg = old_dir->_[left];
    auto right_seg = old_dir->_[right];
     //验证段
    if ((reinterpret_cast<uint64_t>(left_seg) & headerMask) != crash_version) {
      recoverTable(&old_dir->_[left], key_hash, left, old_dir);
      continue;
    }
    //访问段
    if ((reinterpret_cast<uint64_t>(right_seg) & headerMask) != crash_version) {
      recoverTable(&old_dir->_[right], key_hash, right, old_dir);
      continue;
    }
    //新旧模式的pattern
    size_t _pattern0 =
        ((key_hash >> (8 * sizeof(key_hash) - target->local_depth + 1)) << 1);
    size_t _pattern1 =
        ((key_hash >> (8 * sizeof(key_hash) - target->local_depth + 1)) << 1) +
        1;

    /* Get the lock from left to right*/
    if (left_seg->Acquire_and_verify(_pattern0)) {
      if (right_seg->Acquire_and_verify(_pattern1)) {
        //如果左边段的本地深度不等于右边段的本地深度，说明合并完成了
        if (left_seg->local_depth != right_seg->local_depth) {
          left_seg->bucket->release_lock();
          right_seg->bucket->release_lock();
          return;
        }
        //两边的版本号都不为0，说明完成了
        if ((left_seg->number != 0) && (right_seg->number != 0)) {
          left_seg->bucket->release_lock();
          right_seg->bucket->release_lock();
          return;
        }
        //将所有桶锁住
        left_seg->Acquire_remaining_locks();
        right_seg->Acquire_remaining_locks();

        /*First improve the local depth, */
        //减少两边的本地深度，从左到右更新，并且修改
        left_seg->local_depth = left_seg->local_depth - 1;
        Allocator::Persist(&left_seg->local_depth, sizeof(uint64_t));
        left_seg->state = -1;
        Allocator::Persist(&left_seg->state, sizeof(int));
        right_seg->state = -1;
        Allocator::Persist(&right_seg->state, sizeof(int));
      REINSERT:
        old_dir = dir;
        /*Update the directory from left to right*/
        while (Test_Directory_Lock_Set()) {
          asm("nop");
        }
        /*start the merge operation*/
        Directory_Merge_Update(old_dir, key_hash, left_seg);

        if (Test_Directory_Lock_Set() || old_dir->version != dir->version) {
          goto REINSERT;
        }

        if (right_seg->number != 0) {
          left_seg->Merge(right_seg);
        }
        auto reserve_item = Allocator::ReserveItem();
        TX_BEGIN(pool_addr) {
          pmemobj_tx_add_range_direct(reserve_item, sizeof(*reserve_item));
          pmemobj_tx_add_range_direct(&left_seg->next, sizeof(left_seg->next));
          Allocator::Free(reserve_item, right_seg);
          left_seg->next = right_seg->next;
        }
        TX_ONABORT { std::cout << "Error for merge txn" << std::endl; }
        TX_END

        left_seg->pattern = left_seg->pattern >> 1;
        Allocator::Persist(&left_seg->pattern, sizeof(uint64_t));
        left_seg->state = 0;
        Allocator::Persist(&left_seg->state, sizeof(int));
        right_seg->Release_all_locks();
        left_seg->Release_all_locks();
        /*Try to halve directory?*/
        if ((dir->depth_count == 0) && (dir->global_depth > 2)) {
          Lock_Directory();
          if (dir->depth_count == 0) {
            Halve_Directory();
          }
          Unlock_Directory();
        }
      } else {
        //如果右边合并好了的话，释放右边的锁
        left_seg->bucket->release_lock();
        if (old_dir == dir) {
          return;
        }
      }
    } else {
      //pattern不用变得话 返回即可
      if (old_dir == dir) {
        /* If the directory itself does not change, directly return*/
        return;
      }
    }

  } while (true);
}

template <class T>
bool Finger_EH<T>::Delete(T key, bool is_in_epoch) {
  if (!is_in_epoch) {
    auto epoch_guard = Allocator::AquireEpochGuard();
    return Delete(key);
  }
  return Delete(key);
}

/*By default, the merge operation is disabled*/
template <class T>
bool Finger_EH<T>::Delete(T key) {
  /*Basic delete operation and merge operation*/
  uint64_t key_hash;
  //计算key的hash值
  if constexpr (std::is_pointer_v<T>) {
    key_hash = h(key->key, key->length);
  } else {
    key_hash = h(&key, sizeof(key));
  }
  //fingerprint
  auto meta_hash = ((uint8_t)(key_hash & kMask));  // the last 8 bits
RETRY:
  //访问一个段之前需要验证
  auto old_sa = dir;
  //得到段的标号
  auto x = (key_hash >> (8 * sizeof(key_hash) - old_sa->global_depth));
  //dir_entry对应目录的条目由
  auto dir_entry = old_sa->_;
  //得到指向target_table的指针，后56位充当指针，前8位用来标识是否崩溃了
  Table<T> *target_table = reinterpret_cast<Table<T> *>(
      reinterpret_cast<uint64_t>(dir_entry[x]) & tailMask);

  if ((reinterpret_cast<uint64_t>(dir_entry[x]) & headerMask) !=
      crash_version) {
    recoverTable(&dir_entry[x], key_hash, x, old_sa);
    goto RETRY;
  }

  /*we need to first do the locking and then do the verify*/
  //锁住需要删除的桶
  auto y = BUCKET_INDEX(key_hash);
  Bucket<T> *target = target_table->bucket + y;
  Bucket<T> *neighbor = target_table->bucket + ((y + 1) & bucketMask);
  target->get_lock();
  if (!neighbor->try_get_lock()) {
    target->release_lock();
    goto RETRY;
  }
  //old_sa等于现在的dir
  //再验证一遍段
  old_sa = dir;
  x = (key_hash >> (8 * sizeof(key_hash) - old_sa->global_depth));
  if (reinterpret_cast<Table<T> *>(reinterpret_cast<uint64_t>(old_sa->_[x]) &
                                   tailMask) != target_table) {
    //如果不对的话 释放锁
    target->release_lock();
    neighbor->release_lock();
    goto RETRY;
  }
  //删除key
  auto ret = target->Delete(key, meta_hash, false);
  //如果ret==0说明删除成功了
  if (ret == 0) {
#ifdef COUNTING
    auto num = SUB(&target_table->number, 1);
#endif
    target->release_lock();
#ifdef PMEM
    Allocator::Persist(&target->bitmap, sizeof(target->bitmap));
#endif
    neighbor->release_lock();
#ifdef COUNTING
    if (num == 0) {
      TryMerge(key_hash);
    }
#endif
    return true;
  }
  //看看neighbor有没有
  ret = neighbor->Delete(key, meta_hash, true);
  if (ret == 0) {
#ifdef COUNTING
    auto num = SUB(&target_table->number, 1);
#endif
    neighbor->release_lock();
#ifdef PMEM
    Allocator::Persist(&neighbor->bitmap, sizeof(neighbor->bitmap));
#endif
    target->release_lock();
#ifdef COUNTING
    if (num == 0) {
      TryMerge(key_hash);
    }
#endif
    return true;
  }
  //如果target和neighbor都没有的话
  //1.如果当前桶保存了指纹的话，就不需要遍历所有stash桶了
  //2.如果没有的话，就需要遍历所有的stash桶了
  if (target->test_stash_check()) {
    auto test_stash = false;
    //看看要不要查找stash
    if (target->test_overflow()) {
      /*this only occur when the bucket has more key-values than 10 that are
       * overfloed int he shared bucket area, therefore it needs to search in
       * the extra bucket*/
      test_stash = true;
    }  //如果没有溢出到stash的话，说明overflow保存了所有的溢出指纹，比较指纹即可
    else {
      /*search in the original bucket*/
      //mask 溢出指纹的bitmap
      int mask = target->overflowBitmap & overflowBitmapMask;
      //如果mask不等于0的话，遍历一下指纹
      if (mask != 0) {
        for (int i = 0; i < 4; ++i) {
          //这个位置有数据，指纹也相等，并且属于target，如果是b-1溢出的话，那可能就不需要管了
          if (CHECK_BIT(mask, i) &&
              (target->finger_array[14 + i] == meta_hash) &&
              (((1 << i) & target->overflowMember) == 0)) {
            test_stash = true;
            //找到的话，就需要去对应的stash中删除
            goto TEST_STASH;
          }
        }
      }
      //查看neighbor的mask，这时候membership要为1
      mask = neighbor->overflowBitmap & overflowBitmapMask;
      if (mask != 0) {
        for (int i = 0; i < 4; ++i) {
          if (CHECK_BIT(mask, i) &&
              (neighbor->finger_array[14 + i] == meta_hash) &&
              (((1 << i) & neighbor->overflowMember) != 0)) {
            test_stash = true;
            break;
          }
        }
      }
    }

  TEST_STASH:
    if (test_stash == true) {
      Bucket<T> *stash = target_table->bucket + kNumBucket;
      stash->get_lock();
      for (int i = 0; i < stashBucket; ++i) {
        int index = ((i + (y & stashMask)) & stashMask);
        Bucket<T> *curr_stash = target_table->bucket + kNumBucket + index;
        auto ret = curr_stash->Delete(key, meta_hash, false);
        if (ret == 0) {
          /*need to unset indicator in original bucket*/
          stash->release_lock();
#ifdef PMEM
          Allocator::Persist(&curr_stash->bitmap, sizeof(curr_stash->bitmap));
#endif
          auto bucket_ix = BUCKET_INDEX(key_hash);
          auto org_bucket = target_table->bucket + bucket_ix;
          assert(org_bucket == target);
          target->unset_indicator(meta_hash, neighbor, key, index);
#ifdef COUNTING
          auto num = SUB(&target_table->number, 1);
#endif
          neighbor->release_lock();
          target->release_lock();
#ifdef COUNTING
          if (num == 0) {
            TryMerge(key_hash);
          }
#endif
          return true;
        }
      }
      stash->release_lock();
    }
  }
  //释放锁，返回错误
  neighbor->release_lock();
  target->release_lock();
  return false;
}

/*DEBUG FUNCTION: search the position of the key in this table and print
 * correspongdign informantion in this table, to test whether it is correct*/

template <class T>
int Finger_EH<T>::FindAnyway(T key) {
  uint64_t key_hash;
  if constexpr (std::is_pointer_v<T>) {
    // key_hash = h(key, (reinterpret_cast<string_key *>(key))->length);
    key_hash = h(key->key, key->length);
  } else {
    key_hash = h(&key, sizeof(key));
  }
  auto meta_hash = ((uint8_t)(key_hash & kMask));
  auto x = (key_hash >> (8 * sizeof(key_hash) - dir->global_depth));

  size_t _count = 0;
  size_t seg_count = 0;
  Directory<T> *seg = dir;
  Table<T> **dir_entry = seg->_;
  Table<T> *ss;
  auto global_depth = seg->global_depth;
  size_t depth_diff;
  int capacity = pow(2, global_depth);
  for (int i = 0; i < capacity;) {
    ss = dir_entry[i];
    Bucket<T> *curr_bucket;
    for (int j = 0; j < kNumBucket; ++j) {
      curr_bucket = ss->bucket + j;
      auto ret = curr_bucket->check_and_get(meta_hash, key, false);
      if (ret != NONE) {
        printf("successfully find in the normal bucket with false\n");
        printf(
            "the segment is %d, the bucket is %d, the local depth = %lld, the "
            "pattern is %lld\n",
            i, j, ss->local_depth, ss->pattern);
        return 0;
      }
      ret = curr_bucket->check_and_get(meta_hash, key, true);
      if (ret != NONE) {
        printf("successfully find in the normal bucket with true\n");
        printf(
            "the segment is %d, the bucket is %d, the local depth is %lld, the "
            "pattern is %lld\n",
            i, j, ss->local_depth, ss->pattern);
        return 0;
      }
    }

    for (int i = 0; i < stashBucket; ++i) {
      curr_bucket = ss->bucket + kNumBucket + i;
      auto ret = curr_bucket->check_and_get(meta_hash, key, false);
      if (ret != NONE) {
        printf("successfully find in the stash bucket\n");
        auto bucket_ix = BUCKET_INDEX(key_hash);
        auto org_bucket = ss->bucket + bucket_ix;
        auto neighbor_bucket = ss->bucket + ((bucket_ix + 1) & bucketMask);
        printf("the segment number is %d, the bucket_ix is %d\n", x, bucket_ix);

        printf("the image of org_bucket\n");
        int mask = org_bucket->overflowBitmap & overflowBitmapMask;
        for (int j = 0; j < 4; ++j) {
          printf(
              "the hash is %d, the pos bit is %d, the alloc bit is %d, the "
              "stash bucket info is %d, the real stash bucket info is %d\n",
              org_bucket->finger_array[14 + j],
              (org_bucket->overflowMember >> (j)) & 1,
              (org_bucket->overflowBitmap >> j) & 1,
              (org_bucket->overflowIndex >> (j * 2)) & stashMask, i);
        }

        printf("the image of the neighbor bucket\n");
        printf("the stash check is %d\n", neighbor_bucket->test_stash_check());
        mask = neighbor_bucket->overflowBitmap & overflowBitmapMask;
        for (int j = 0; j < 4; ++j) {
          printf(
              "the hash is %d, the pos bit is %d, the alloc bit is %d, the "
              "stash bucket info is %d, the real stash bucket info is %d\n",
              neighbor_bucket->finger_array[14 + j],
              (neighbor_bucket->overflowMember >> (j)) & 1,
              (neighbor_bucket->overflowBitmap >> j) & 1,
              (neighbor_bucket->overflowIndex >> (j * 2)) & stashMask, i);
        }

        if (org_bucket->test_overflow()) {
          printf("the org bucket has overflowed\n");
        }
        return 0;
      }
    }

    depth_diff = global_depth - ss->local_depth;
    _count += ss->number;
    seg_count++;
    i += pow(2, depth_diff);
  }
  return -1;
}

#undef BUCKET_INDEX
#undef GET_COUNT
#undef GET_BITMAP
#undef GET_MEMBER
#undef GET_INVERSE_MEMBER
}  // namespace extendible
