import java.util.ArrayList;

public class Server implements Subject {
    ArrayList<Observer> observers;
    Object data;

    Server(Iterator iterator){
        while(!iterator.isDone()){
            Observer observer = (Observer)iterator.next();
            registerObserver(observer);
        }
    }

    @Override
    public void registerObserver(Observer observer) {
        observers.add(observer);
    }

    @Override
    public void unregisterObserver(Observer observer) {
        observers.remove(observer);
    }

    @Override
    public void notifyObservers() {
        observers.forEach(observer -> observer.update(this, data));
    }

    @Override
    public void setData(Object data) {
        this.data = data;
        notifyObservers();
    }
}
