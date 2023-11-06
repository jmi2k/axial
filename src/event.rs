use std::collections::{HashMap, HashSet};

use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, KeyEvent, WindowEvent},
    keyboard::{KeyCode, PhysicalKey},
};

use crate::types::Direction;

#[derive(Eq, PartialEq, Hash)]
pub enum Input {
    Close,
    Focus,
    Unfocus,
    Motion,
    Press(KeyCode),
    Release(KeyCode),
}

#[derive(Copy, Clone, Default)]
pub enum Action {
    #[default]
    Nop,

    Exit,
    Fullscreen,
    Turn,
    Walk(Direction),
    Stop(Direction),

    Redraw,
    Resize(PhysicalSize<u32>),

    #[allow(unused)]
    Debug(&'static str),
}

pub struct Handler {
    pub bindings: HashMap<Input, Action>,
    pub pressed_keys: HashSet<KeyCode>,
    pub delta: (f64, f64),
}

impl Handler {
    fn lookup_binding(&self, input: Input) -> Action {
        self.bindings.get(&input).copied().unwrap_or_default()
    }

    pub fn handle(&mut self, event: Event<()>) -> Action {
        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => self.lookup_binding(Input::Close),

            Event::WindowEvent {
                event: WindowEvent::Focused(focus),
                ..
            } => self.lookup_binding(if focus { Input::Focus } else { Input::Unfocus }),

            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                self.delta = delta;
                self.lookup_binding(Input::Motion)
            }

            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { event, .. },
                ..
            } => self.handle_key(event),

            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => Action::Resize(size),

            Event::AboutToWait => Action::Redraw,
            _ => Action::default(),
        }
    }

    fn handle_key(&mut self, event: KeyEvent) -> Action {
        let KeyEvent {
            physical_key: PhysicalKey::Code(code),
            state,
            ..
        } = event
        else {
            return Action::default();
        };

        let pressed = self.pressed_keys.contains(&code);

        match state {
            ElementState::Pressed if !pressed => {
                self.pressed_keys.insert(code);
                self.lookup_binding(Input::Press(code))
            }

            ElementState::Released if pressed => {
                self.pressed_keys.remove(&code);
                self.lookup_binding(Input::Release(code))
            }

            _ => Action::default(),
        }
    }
}

impl<T> From<T> for Handler
where
    T: Into<HashMap<Input, Action>>,
{
    fn from(value: T) -> Self {
        Self {
            bindings: value.into(),
            pressed_keys: HashSet::new(),
            delta: Default::default(),
        }
    }
}
