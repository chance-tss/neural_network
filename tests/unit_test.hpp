#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <cmath>

namespace unit_test {

struct Test {
    std::string name;
    std::function<void()> func;
};

inline std::vector<Test>& getTests() {
    static std::vector<Test> tests;
    return tests;
}

struct TestRegistrar {
    TestRegistrar(const std::string& name, std::function<void()> func) {
        getTests().push_back({name, func});
    }
};

inline int runTests() {
    int passed = 0;
    int failed = 0;
    for (const auto& test : getTests()) {
        try {
            test.func();
            std::cout << "[PASS] " << test.name << std::endl;
            passed++;
        } catch (const std::exception& e) {
            std::cout << "[FAIL] " << test.name << ": " << e.what() << std::endl;
            failed++;
        } catch (...) {
            std::cout << "[FAIL] " << test.name << ": Unknown error" << std::endl;
            failed++;
        }
    }
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "Tests passed: " << passed << ", failed: " << failed << std::endl;
    return failed > 0 ? 1 : 0;
}

class TestException : public std::exception {
    std::string msg;
public:
    TestException(const std::string& m) : msg(m) {}
    const char* what() const noexcept override { return msg.c_str(); }
};

} // namespace unit_test

#define TEST(name) \
    void name(); \
    static unit_test::TestRegistrar registrar_##name(#name, name); \
    void name()

#define ASSERT_TRUE(condition) \
    if (!(condition)) throw unit_test::TestException("Assertion failed: " #condition);

#define ASSERT_EQ(a, b) \
    if ((a) != (b)) throw unit_test::TestException("Assertion failed: " #a " == " #b);

#define ASSERT_NEAR(a, b, epsilon) \
    if (std::abs((a) - (b)) > (epsilon)) throw unit_test::TestException("Assertion failed: " #a " near " #b);
