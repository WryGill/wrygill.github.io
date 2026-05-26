---
title: ArrayList和Vector底层初探
date: 2025-09-26 12:52:43
tags: ["集合", "底层"]
categories: [JavaSE]
---

# ArrayList和Vector底层初探
## ArrayList
`ArrayList` 实现于 `List`、`RandomAccess` 接口。可以插入空数据，也支持随机访问。
`ArrayList`有两个重要属性：
  * Object[] : elementData
  * int : size


### add方法
  * 默认**add(E e)** ：
        public boolean add(E e) {
            // 1. 确保容量足够
            ensureCapacityInternal(size + 1);  
            // 2. 将元素添加到数组末尾
            elementData[size++] = e; 
            return true;
        }
确保容量足够 -> 将元素添加到数组末尾 -> 尺寸递增
  * 指定下标**add(int index, E e)** ：
        public void add(int index, E element) {
            rangeCheckForAdd(index);
          
            ensureCapacityInternal(size + 1);  // Increments modCount!!
            //复制，向后移动
            System.arraycopy(elementData, index, elementData, index + 1,
                             size - index);
            elementData[index] = element;
            size++;
        }
使用`System.arraycopy`方法，先将目标下标及其后面的元素拷贝到index+1的位置，再在index添加元素


### 扩容机制
    
    // ArrayList 源码中的扩容逻辑
    private void ensureCapacityInternal(int minCapacity) {
        // 如果需要的容量超过了当前数组的容量
        if (minCapacity - elementData.length > 0) {
            // 调用 grow 方法进行扩容
            grow(minCapacity);
        }
    }
    
    private void grow(int minCapacity) {
        int oldCapacity = elementData.length;
        // 计算新容量，这里体现了 1.5 倍扩容的逻辑
        int newCapacity = oldCapacity + (oldCapacity >> 1);
    
        // 如果新容量仍然不够，直接使用需要的最小容量
        if (newCapacity - minCapacity < 0) {
            newCapacity = minCapacity;
        }
    
        // 复制数组，这是最耗时的操作
        elementData = Arrays.copyOf(elementData, newCapacity);
    }
ArrayList的扩容一般是1.5倍，如果扩容后空间仍不足，则使用最小所需空间
## Vector
Vector和ArrayList大差不差，但是Vector相对于ArrayList是**线程安全** 的
### add方法
    
    public synchronized boolean add(E e) {
        modCount++;
        ensureCapacityHelper(elementCount + 1);
        elementData[elementCount++] = e;
        return true;
    }
相比于ArrayList，Vector的add方法采用了`synchronized`关键字修饰，确保只有一个线程能够访问Vector对象，但是开销较大。
### 扩容机制
    
    // 这是 Vector 源码中的扩容方法
    private void grow(int minCapacity) {
        int oldCapacity = elementData.length;
        // 这里计算新容量，如果 capacityIncrement > 0，则按增量增长，否则直接翻倍
        int newCapacity = oldCapacity + ((capacityIncrement > 0) ? capacityIncrement : oldCapacity);
        // ... 后续逻辑和 ArrayList 类似，进行数组复制 ...
        elementData = Arrays.copyOf(elementData, newCapacity);
    }
Vector的扩容是一次扩容2倍，这点与ArrayList不同
## 面试题
### ArrayList和Vector有什么不同？
  * ArrayList是非线程安全的，Vector是线程安全的
  * ArrayList在单线程环境下效率比Vector要高
  * 扩容时，ArrayList是1.5倍，Vector是2倍


### 为什么不推荐使用Vector？
  * 粒度太粗、性能太低：
    * vector的方法都采用synchronized关键字修饰，在多线程下会导致**所有线程都必须排队** 访问，锁竞争非常严重，性能比 ArrayList 差很多。
  * 有更好的方案：
    * **CopyOnWriteArrayList** ：适合读多写少的场景
    * **ConcurrentLinkedQueue** ：适合高并发队列场景