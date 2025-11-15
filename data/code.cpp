#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <map>
#include <algorithm>
using namespace std;

// A classic-style C++ program with various utilities
// Not modern C++17/20 style — intentionally old-school

class Record {
public:
    int id;
    string name;
    double value;

    Record(): id(0), value(0.0) {}
    Record(int i, const string &n, double v): id(i), name(n), value(v) {}

    void print() const {
        cout << "Record[" << id << "] " << name << " = " << value << endl;
    }
};

class Database {
private:
    vector<Record> records;

public:
    void addRecord(const Record &r) {
        records.push_back(r);
    }

    Record *getRecordById(int id) {
        for (size_t i = 0; i < records.size(); ++i) {
            if (records[i].id == id)
                return &records[i];
        }
        return NULL;
    }

    void sortByName() {
        sort(records.begin(), records.end(),
            [](const Record &a, const Record &b) {
                return a.name < b.name;
            }
        );
    }

    void sortByValue() {
        sort(records.begin(), records.end(),
            [](const Record &a, const Record &b) {
                return a.value < b.value;
            }
        );
    }

    void printAll() const {
        for (size_t i = 0; i < records.size(); ++i)
            records[i].print();
    }
};

vector<string> split(const string &s, char sep) {
    vector<string> out;
    string current;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == sep) {
            out.push_back(current);
            current.clear();
        } else {
            current += s[i];
        }
    }
    out.push_back(current);
    return out;
}

int main(int argc, char **argv) {
    cout << "Classic C++ Example Program" << endl;

    if (argc < 2) {
        cout << "Usage: ./a.out <input-file>" << endl;
        return 1;
    }

    ifstream fin(argv[1]);
    if (!fin.is_open()) {
        cout << "Could not open file: " << argv[1] << endl;
        return 1;
    }

    Database db;
    string line;

    while (getline(fin, line)) {
        if (line.size() == 0)
            continue;
        vector<string> parts = split(line, ',');
        if (parts.size() < 3)
            continue;

        int id = atoi(parts[0].c_str());
        string name = parts[1];
        double val = atof(parts[2].c_str());

        db.addRecord(Record(id, name, val));
    }

    cout << "Loaded records:" << endl;
    db.printAll();

    cout << "\nSorting by name..." << endl;
    db.sortByName();
    db.printAll();

    cout << "\nSorting by value..." << endl;
    db.sortByValue();
    db.printAll();

    cout << "Done." << endl;
    return 0;
}
