use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use std::sync::Arc;
use rand::Rng;
use std::hash::Hash;
// use std::rc::Rc;

// A node in the graph
struct Node {
    // The index of the node
    index: usize,
    // The distance from the source node
    distance: usize,
}

// Implement `PartialEq` and `Eq` so we can use `Node` in a `BinaryHeap`
impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Node {}

// Implement `PartialOrd` and `Ord` so we can use `Node` in a `BinaryHeap`
impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.cmp(&self.distance)
    }
}

// The graph represented as an adjacency list
struct Graph {
    adjacency_list: Vec<Vec<(usize, usize)>>,
}

impl Graph {
    fn new(n: usize) -> Self {
        Self { adjacency_list: vec![Vec::new(); n] }
    }

    fn add_edge(&mut self, from: usize, to: usize, weight: usize) {
        self.adjacency_list[from].push((to, weight));
    }

    // Calculate the shortest path from the source node to all other nodes
    fn shortest_paths(&self, source: usize) -> Vec<usize> {
        // Initialize the distances and previous nodes
        let mut distances: Vec<_> = (0..self.adjacency_list.len()).map(|_| usize::max_value()).collect();
        let mut previous: Vec<_> = (0..self.adjacency_list.len()).map(|_| None).collect();

        // Set the distance of the source node to 0
        distances[source] = 0;

        // Use a binary heap to efficiently find the node with the smallest distance
        let mut heap = BinaryHeap::new();
        heap.push(Node { index: source, distance: 0 });

        // The set of visited nodes
        let mut visited = HashMap::new();

        // Process each node in the heap
        while let Some(Node { index, distance }) = heap.pop() {
            // Skip this node if it has already been visited
            if visited.contains_key(&index) {
                continue;
            }

            // Mark the node as visited
            visited.insert(index, true);

            // Update the distances of the neighbors
            for &(next, weight) in &self.adjacency_list[index] {
                let next_distance = distance + weight;
                if next_distance < distances[next] {
                    distances[next] = next_distance;
                    previous[next] = Some(index);
                    heap.push(Node { index: next, distance: next_distance });
                }
            }
        }

        distances
    }
}

const N: u64 = 100_000_000;

#[derive(Clone, Debug)]
enum Test {
    String(String),
    Number(usize),
    Empty,
}

impl Default for Test {
    fn default() -> Self {
        Test::Empty
    }
}

pub struct AssociativeArray<K: std::hash::Hash, V> {
    // pub key: Vec<K>,
    pub val: Vec<V>,
    pub key_to_index: HashMap<K, usize>,
    pub dead: Vec<bool>,
    pub dead_count: usize,
}

struct AssociativeArrayReorg {
    pub old_index_to_new: Vec<usize>,
}

impl AssociativeArrayReorg {
    pub fn get_new_index(&self, index: usize) -> Option<usize> {
        if self.is_dead(index) {
            None
        } else {
            Some(self.old_index_to_new[index])
        }
    }

    pub fn is_dead(&self, index: usize) -> bool {
        self.old_index_to_new[index] == usize::MAX
    }
}

impl<K: Hash + PartialEq + Eq, V> AssociativeArray<K, V> {
    pub fn new() -> Self {
        Self {
            val: vec![],
            key_to_index: HashMap::new(),
            dead: vec![],
            dead_count: 0,
        }
    }
    
    pub fn insert(&mut self, key: K, val: V) -> usize {
        let index = self.val.len();
        // self.key.push(key);
        self.val.push(val);
        self.key_to_index.insert(key, index);
        self.dead.push(false);

        return index;
    }

    pub fn delete_dont_reorg(&mut self, key: &K) {
        let index = self.key_to_index.remove(key).unwrap();
        self.dead[index] = true;
        self.dead_count += 1;
    }

    pub fn delete(&mut self, key: &K) {
        self.delete_dont_reorg(key);

        self.reorg_if_half_dead();
    }

    pub fn reorg_if_half_dead(&mut self) -> Option<AssociativeArrayReorg> {
        if self.dead.len() < self.dead_count * 2 {
            Some(self.reorg())
        } else {
            None
        }
    }

    pub fn reorg(&mut self) -> AssociativeArrayReorg {
        let mut reorg = AssociativeArrayReorg {
            old_index_to_new: vec![usize::MAX; self.dead.len()],
        };

        let mut pull_from_offset = 0;
        for i in 0..self.val.len() {
            if self.dead[i] {
                pull_from_offset += 1;
            }

            self.dead[i] = false;

            if 0 < pull_from_offset {
                self.val.swap(i, i + pull_from_offset);
            }

            reorg.old_index_to_new[i + pull_from_offset] = i;
        }

        let new_len = self.val.len() - self.dead_count;
        // self.key.truncate(new_len);
        self.val.truncate(new_len);
        self.dead.truncate(new_len);
        self.dead_count = 0;

        return reorg
    }
}

fn main() {
    struct A(Vec<Arc<A>>);

    // let mut a1 = Rc::new(A(vec![]));
    // let mut a2 = Rc::new(A(vec![]));

    // a1.0.push(a2.clone());
    // a2.0.push(a1.clone());
    
    // let x = Arc::new(0);
    // let y = x.clone();
    // println!("{:?}", x == y);

    // use std::sync::atomic::{AtomicUsize, Ordering};

    // let mut rng = rand::thread_rng();
    // let bools: Vec<bool> = (0..N).map(|_| rng.gen()).collect();
    // let mut count = 0;
    // let atom = AtomicUsize::new(0);
    // let xbeam: crossbeam::atomic::AtomicCell<usize> = crossbeam::atomic::AtomicCell::new(0);

    // let start = chrono::Utc::now();
    // for i in 1..N { if i % 2 == 0 { count += 1; } }
    // println!("{:?} {:?}", count, chrono::Utc::now() - start);

    // // TODO: how fast are references?

    // let start = chrono::Utc::now();
    // for i in 1..N { if i % 2 == 0 { atom.fetch_add(1, Ordering::Relaxed); }}
    // println!("{:?} {:?}", atom, chrono::Utc::now() - start);

    // let start = chrono::Utc::now();
    // for i in 1..N { if i % 20 != 0 { xbeam.fetch_add(1); }}
    // println!("{:?} {:?}", xbeam, chrono::Utc::now() - start);

    // let start = chrono::Utc::now();
    // for b in &bools { if *b { count += 1; } }
    // println!("{:?} {:?}", count, chrono::Utc::now() - start);

    // let start = chrono::Utc::now();
    // for b in &bools { if *b { count += 1; } }
    // println!("{:?} {:?}", count, chrono::Utc::now() - start);

    // let mut test: Vec<crossbeam::atomic::AtomicCell<Test>> = vec![];
    // for _ in 0..N {
    //     test.push(crossbeam::atomic::AtomicCell::new(Test::Number(0)));
    // }

    let mut rng = rand::thread_rng();
    let mut random_numbers = vec![];
    for _ in 0..N {
        random_numbers.push(rng.gen_range(0..N) as usize);
    }
    
    // let start = chrono::Utc::now();
    // for i in random_numbers.iter() {
    //     match test[*i].take() {
    //         Test::Number(n) => {
    //             test[*i].store(Test::Number(n + 1));
    //         },
    //         _ => panic!("bad"),
    //     }
    // }
    // println!("write items {:?}", chrono::Utc::now() - start);

    // let start = chrono::Utc::now();
    // let mut total = 0;
    // for atom in test.iter() {
    //     match atom.take() {
    //         Test::Number(n) => {
    //             total += n;
    //         },
    //         _ => panic!("bad"),
    //     }
    // }
    // println!("read items {:?} {:?}", total, chrono::Utc::now() - start);

    // let start = chrono::Utc::now();
    // let total = Arc::new(std::sync::RwLock::new(0));
    // for i in random_numbers.iter() {
    //     let mut x = total.write().unwrap();
    //     *x += 1;
    // }
    // println!("{:?} {:?}", total, chrono::Utc::now() - start);

    // let start = chrono::Utc::now();
    // println!("about to create rwlocks");
    // let locks = (0..100_000_000).map(|i| Arc::new(std::sync::RwLock::new(i % 20))).collect::<Vec<Arc<std::sync::RwLock<usize>>>>();
    // println!("create rwlocks {:?}", chrono::Utc::now() - start);
    // let start = chrono::Utc::now();
    // let mut total = 0;
    // for lock in locks.iter() {
    //     total += (*lock.read().unwrap());
    // }
    // println!("read rwlocks {:?} {:?}", total, chrono::Utc::now() - start);

    // let start = chrono::Utc::now();
    // println!("about to create rwlocks");
    // let locks = (0..100_000_000).map(|i| Arc::new(parking_lot::RwLock::new(i % 20))).collect::<Vec<Arc<parking_lot::RwLock<usize>>>>();
    // println!("create rwlocks {:?}", chrono::Utc::now() - start);
    // let start = chrono::Utc::now();
    // let mut total = 0;
    // for lock in locks.iter() {
    //     total += (*lock.read());
    // }
    // println!("read rwlocks {:?} {:?}", total, chrono::Utc::now() - start);

    // let start = chrono::Utc::now();
    // let locks = (0..10_000_000).map(|i| Arc::new(std::sync::RwLock::new(i))).collect::<Vec<Arc<std::sync::RwLock<usize>>>>();
    // println!("create locks {:?}", chrono::Utc::now() - start);
    
    // let start = chrono::Utc::now();
    // let mut total = 0;
    // for lock in locks.iter() {
    //     let mut x = lock.write().unwrap();
    //     *x += 1;
    // }
    // println!("read locks {:?}", chrono::Utc::now() - start);
    
    // let start = chrono::Utc::now();
    // let mut total = 0;
    // for lock in locks.iter() {
    //     total += (*lock.read().unwrap()) % 20;
    // }
    // println!("{:?} {:?}", total, chrono::Utc::now() - start);    

    // let start = chrono::Utc::now();
    // let locks = (0..10_000_000).map(|i| Arc::new(std::sync::Mutex::new(Test::Number(i % 20)))).collect::<Vec<Arc<std::sync::Mutex<Test>>>>();
    // println!("create locks {:?}", chrono::Utc::now() - start);
    
    // let start = chrono::Utc::now();
    // let mut total = 0;
    // for lock in locks.iter() {
    //     match *lock.lock().unwrap() {
    //         Test::Number(n) => {
    //             total += n;
    //         },
    //         _ => panic!("bad"),
    //     }
    // }
    // println!("read locks {total:?} {:?}", chrono::Utc::now() - start);
    
    let start = chrono::Utc::now();
    let locks = (0..10_000_000).map(|i| Arc::new(parking_lot::Mutex::new(Test::Number(i % 20)))).collect::<Vec<Arc<parking_lot::Mutex<Test>>>>();
    println!("create locks {:?}", chrono::Utc::now() - start);
    
    let start = chrono::Utc::now();
    let mut total = 0;
    for lock in locks.iter() {
        match *lock.lock().unwrap() {
            Test::Number(n) => {
                total += n;
            },
            _ => panic!("bad"),
        }
    }
    println!("read locks {total:?} {:?}", chrono::Utc::now() - start);

    // let start = chrono::Utc::now();
    // let mut total = std::sync::Mutex::new(0);
    // for number in random_numbers.iter() {
    //     let mut x = total.lock().unwrap();
    //     *x += number % 20;
    // }
    // println!("with mutex {:?} {:?}", total, chrono::Utc::now() - start);    

    // let start = chrono::Utc::now();
    // let mut total = 0;
    // for lock in locks.iter() {
    //     total += (*lock.lock().unwrap()) % 20;
    // }
    // println!("{:?} {:?}", total, chrono::Utc::now() - start);

    // // let x = std::sync::Mutex::new(Test::String("asdf".to_string()));
    // // let mut y = x.lock();
    // // *y = Test::String("asdf".to_string());

    // let start = chrono::Utc::now();
    // let mut total = core::sync::atomic::AtomicUsize::new(0);
    // for number in random_numbers.iter() {
    //     total.fetch_add(number % 20, std::sync::atomic::Ordering::Relaxed);
    // }
    // println!("with atomic usize {:?} {:?}", total, chrono::Utc::now() - start);

    // let mut arena = generational_arena::Arena::new();
    // let mut total = arena.insert(Test::Number(0));
    // let mut ids = vec![];
    // let start = chrono::Utc::now();
    // for (number, i) in random_numbers.iter().enumerate() {
    //     let mut val = arena.get(total).unwrap();
    //     match *val {
    //         Test::Number(x) => {
    //             ids.push(total);
    //             total = arena.insert(Test::Number(x + (number % 20)));
    //         }
    //         _ => panic!("bad"),
    //     }        
    // }
    // println!("with arena {:?} {:?}", arena.get(total).unwrap(), chrono::Utc::now() - start);

    // let mut total = 0;
    // let start = chrono::Utc::now();
    // for id in ids.iter() {
    //     let mut val = arena.get(*id).unwrap();
    //     match val {
    //         Test::Number(x) => {
    //             total += (x % 20);
    //         }
    //         _ => panic!("bad"),
    //     }        
    // }
    // println!("arena all references {:?} {:?}", total, chrono::Utc::now() - start);

    // let start = chrono::Utc::now();
    // let mut dict = AssociativeArray::new();
    // for n in random_numbers.iter() {
    //     dict.insert(*n, Test::Number(*n));
    // }
    // println!("insert associative array {:?}", chrono::Utc::now() - start);

    // let start = chrono::Utc::now();
    // let mut total = 0;
    // for i in 0..random_numbers.len() {
    //     total += match dict.val[i] {
    //         Test::Number(n) => n,
    //         _ => panic!("bad"),
    //     };
    // }
    // println!("add up associative array {:?} {total}", chrono::Utc::now() - start);

    // let start = chrono::Utc::now();
    // let total = std::cell::Cell::new(0);
    // for number in random_numbers.iter() {
    //     let mut x = total.get()

    // let start = chrono::Utc::now();
    // for b in &bools { if *b { atom.fetch_add(1, Ordering::Relaxed); }}
    // println!("{:?} {:?}", atom, chrono::Utc::now() - start);

    // // Create a graph with 5 nodes
    // let mut graph = Graph::new(5);

    // // Add edges to the graph
    // graph.add_edge(0, 1, 10);
    // graph.add_edge(0, 4, 5);
    // graph.add_edge(1, 2, 1);
    // graph.add_edge(1, 4, 2);
    // graph.add_edge(2, 3, 4);
    // graph.add_edge(3, 2, 6);
    // graph.add_edge(3, 0, 7);
    // graph.add_edge(4, 1, 3);
    // graph.add_edge(4, 2, 9);
    // graph.add_edge(4, 3, 2);

    // // Calculate the shortest paths from node 0
    // let distances = graph.shortest_paths(0);

    // // Print the distances to each node
    // println!("{:?}", distances);
}




