import axios from "axios";
import { toast } from "react-toastify";

export let hasShownLostConnectionToServerToast = false;

const customAxios = axios.create({
  baseURL: import.meta.env.VITE_API_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

customAxios.interceptors.response.use(
  (response) => {
    hasShownLostConnectionToServerToast = false;
    if (response && response.data) {
      return response.data;
    }
  },
  (error) => {
    const { code, message, response } = error;

    if (response && response.data.error && response.data.error_description) {
      toast.error(response.data.error_description);
    } else if (response && response.data.error_description) {
      toast.error(response.data.error_description);
    } else if (response && response.data.error) {
      toast.error(response.data.error);
    }

    if (
      (code === "ERR_NETWORK" || message === "Network Error") &&
      !hasShownLostConnectionToServerToast
    ) {
      toast.error(
        "Mất kết nối với server máy trạm. Vui lòng khởi động lại chương trình !"
      );
      hasShownLostConnectionToServerToast = true;
    }

    return Promise.reject(error);
  }
);

export const resetHasShownLostConnectionToServerToast = () => {
  hasShownLostConnectionToServerToast = false;
};

export default customAxios;
