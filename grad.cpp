#include <cassert>
#include <memory>
#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <functional>
#include <iostream>
#include <cassert>


class Value;

using ValuePtr = std::shared_ptr<Value>;

struct ValHash {
    size_t operator()(const ValuePtr value) const;
};

class Value : public std::enable_shared_from_this<Value> {
public:
    inline static size_t currentID = 0;
    float data;
    float grad;
    std::string op;
    size_t id;
    std::vector<ValuePtr> prev;
    std::function<void()> backward;

private:  //priv constructor
    Value (float data, const std::string &op, size_t id) : data(data), op(op), id(id) {};

public:
    static ValuePtr create(float data, const std::string &op = ""){
        return std::shared_ptr<Value>(new Value(data, op, Value::currentID++));
    }

    ~Value(){
        --Value::currentID;
    }

    static ValuePtr add(const ValuePtr& lhs, const ValuePtr& rhs) {
        ValuePtr out = Value::create(lhs->data + rhs->data, "+");
        out->prev = {lhs, rhs};

        out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), 
                        rhs_weak = std::weak_ptr<Value>(rhs),
                        out_weak = std::weak_ptr<Value>(out)](){
            lhs_weak.lock()->grad += out_weak.lock()->grad;
            rhs_weak.lock()->grad += out_weak.lock()->grad;
        };

        return out;
    }

    static ValuePtr multiply(const ValuePtr& lhs, const ValuePtr& rhs) {
        ValuePtr out = Value::create(lhs->data * rhs->data, "*");
        out->prev = {lhs, rhs};

        out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), 
                        rhs_weak = std::weak_ptr<Value>(rhs),
                        out_weak = std::weak_ptr<Value>(out)](){
            lhs_weak.lock()->grad += rhs_weak.lock()->data * out_weak.lock()->grad;
            rhs_weak.lock()->grad += lhs_weak.lock()->data * out_weak.lock()->grad;
        };

        return out;
    }

    static ValuePtr divide(const ValuePtr& lhs, const ValuePtr& rhs) {
        ValuePtr reciprocal = pow(rhs, -1);
        return multiply(lhs, reciprocal);
    }


    static ValuePtr relu(const ValuePtr& input) {
        float relu_val = std::max(0.0f, input->data);
        auto out = Value::create(relu_val, "ReLU");
        out->prev = {input};

        out->backward = [input, out](){
            //relu gradient is just gradient
            if (input) input->grad += (out->data > 0) * out->grad;
        };

        return out;
    }

    static ValuePtr subtract(const ValuePtr& lhs, const ValuePtr& rhs) {
        ValuePtr out = Value::create(lhs->data - rhs->data, "-");
        out->prev = {lhs, rhs};

        out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), 
                        rhs_weak = std::weak_ptr<Value>(rhs),
                        out_weak = std::weak_ptr<Value>(out)](){
            lhs_weak.lock()->grad += out_weak.lock()->grad;
            rhs_weak.lock()->grad -= out_weak.lock()->grad;
        };

        return out;
    }

    static ValuePtr pow(const ValuePtr& base, float exponent) {
        float new_val = std::pow(base->data, exponent);
        ValuePtr out = Value::create(new_val, "^");
        out->prev = {base};

        out->backward = [base_weak = std::weak_ptr<Value>(out),
                         out_weak = std::weak_ptr<Value>(out),
                         exponent](){

            float grad_factor = exponent * std::pow(base_weak.lock()->data, exponent - 1);
            base_weak.lock()->grad += grad_factor * out_weak.lock()->grad;
        };

        return out;
    }


    void buildTopo(ValuePtr v, std::unordered_set<ValuePtr, ValHash>& visited, std::vector<ValuePtr>& topo){
        if (!visited.count(v)){
            visited.insert(v);
            for (const auto& child : v->prev){
                buildTopo(child, visited, topo);
            }
            topo.push_back(v);
        }
    }
    
    void backProp(){
        this->grad = 1.0f;

        std::vector<ValuePtr> topo;
        std::unordered_set<ValuePtr, ValHash> visited;
        buildTopo(shared_from_this(), visited, topo); //create graph
        
        for (auto it = topo.rbegin(); it != topo.rend(); it++){
            if ((*it)->backward){
                (*it)->backward();
            }
            (*it)->print();
        }
    }

    void print(){
        std::cout << "[data = " << this->data << ", grad = " << this->grad << "]\n";
    }
};

size_t ValHash::operator()(const ValuePtr value) const {
    return std::hash<std::string>()(value.get()->op) ^ std::hash<float>()(value.get()->data);
}

int main() {
    ValuePtr a = Value::create(1.0, "");
    ValuePtr b = Value::create(2.0, "");

    ValuePtr c = Value::add(a, b);
    
    ValuePtr d = Value::multiply(c, c);

    assert(c->data == 3.0);
    assert(c->op == "+");

    assert(d->data == 9.0);
    assert(d->op == "*");

    ValuePtr loss = Value::add(d,d);

    loss->backProp();

};

